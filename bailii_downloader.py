"""Bailii downloader for England and Wales case law courts.

This module provides a polite, resumable crawler that discovers courts from
https://www.bailii.org/databases.html, walks year and month listings, and
exports individual case pages to PDF via Playwright.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.async_api import Browser, BrowserContext, Page, async_playwright


BASE_URL = "https://www.bailii.org/databases.html"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
INVALID_FS_CHARS = r"\\/:*?\"<>|"


def sanitize_for_fs(value: str) -> str:
    """Return a Windows-safe, human-readable string for filesystem usage."""

    sanitized = "".join("-" if ch in INVALID_FS_CHARS else ch for ch in value)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized or "untitled"


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_year_range(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    match = re.match(r"^(\d{4})(?:-(\d{4}))?$", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "--years must be in the form YYYY or YYYY-YYYY"
        )
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if end < start:
        start, end = end, start
    return start, end


def month_from_text(text: str) -> Optional[str]:
    for month in MONTH_NAMES:
        if re.search(rf"\b{re.escape(month)}\b", text, re.IGNORECASE):
            return month
    return None


@dataclass
class CaseRecord:
    court: str
    year: str
    month: str
    title: str
    url: str
    pdf_path: str
    status: str = "pending"
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


class BailiiDownloader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.out).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.jsonl"
        self.csv_path = self.output_dir / "cases.csv"
        self.court_filters = (
            {c.strip().lower() for c in args.courts.split(",")}
            if args.courts.lower() != "all"
            else None
        )
        self.year_range = parse_year_range(args.years)
        self.max_cases = args.max_cases
        self.delay_min = args.delay_min
        self.delay_max = args.delay_max
        self.headless = args.headless
        self.resume = args.resume
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.progress: Dict[str, CaseRecord] = {}
        self.case_records: Dict[str, CaseRecord] = {}
        self.processed_cases = 0
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    # ------------------------------------------------------------------
    # Progress handling
    # ------------------------------------------------------------------
    def load_progress(self) -> None:
        if not self.resume or not self.progress_path.exists():
            logging.info("Starting without existing progress log.")
            return

        with self.progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record = CaseRecord(**data)
                self.progress[record.url] = record
        logging.info("Loaded %d progress entries.", len(self.progress))

    def _append_progress(self, record: CaseRecord) -> None:
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def update_record(self, record: CaseRecord) -> None:
        record.updated_at = time.time()
        self.case_records[record.url] = record
        self.progress[record.url] = record
        self._append_progress(record)

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------
    def polite_delay(self) -> None:
        delay = random.uniform(self.delay_min, self.delay_max)
        logging.debug("Sleeping for %.2f seconds to be polite.", delay)
        time.sleep(delay)

    def _fetch_html_sync(self, url: str) -> str:
        attempts = 0
        backoff = self.delay_min
        while attempts < 5:
            self.polite_delay()
            try:
                response = self.session.get(url, timeout=45)
                if response.status_code in {429} or response.status_code >= 500:
                    logging.warning(
                        "Received status %s for %s, backing off for %.1fs",
                        response.status_code,
                        url,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    attempts += 1
                    continue

                response.raise_for_status()
                text = response.text
                if "excessive traffic" in text.lower():
                    logging.warning(
                        "Potential block detected on %s, backing off for %.1fs",
                        url,
                        backoff,
                    )
                    time.sleep(backoff * 2)
                    attempts += 1
                    continue
                return text
            except requests.RequestException as exc:
                logging.error("Error fetching %s: %s", url, exc)
                time.sleep(backoff)
                backoff *= 2
                attempts += 1
        raise RuntimeError(f"Failed to fetch {url} after retries")

    async def fetch_html(self, url: str) -> str:
        return await asyncio.to_thread(self._fetch_html_sync, url)

    # ------------------------------------------------------------------
    # Playwright lifecycle
    # ------------------------------------------------------------------
    async def ensure_browser(self) -> None:
        if self.browser:
            return
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()

    async def close_browser(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    async def discover_courts(self) -> List[Tuple[str, str]]:
        logging.info("Fetching main database page: %s", BASE_URL)
        html = await self.fetch_html(BASE_URL)
        soup = BeautifulSoup(html, "html.parser")

        def is_england_wales_heading(tag: BeautifulSoup) -> bool:
            if tag.name not in {"h2", "h3", "h4"}:
                return False
            text = tag.get_text(" ", strip=True)
            return bool(
                re.search(r"England\s*&?\s*Wales", text, re.IGNORECASE)
                and re.search(r"Case\s*Law", text, re.IGNORECASE)
            )

        heading = soup.find(is_england_wales_heading)
        if not heading:
            # Fallback: look for a heading that mentions England/Wales even if it omits "Case Law".
            heading = soup.find(
                lambda tag: tag.name in {"h2", "h3", "h4"}
                and re.search(r"England\s*&?\s*Wales", tag.get_text(" ", strip=True), re.IGNORECASE)
            )
        if not heading:
            raise RuntimeError("Could not locate 'England and Wales Case Law' section")

        def find_courts_list() -> Optional[BeautifulSoup]:
            # First preference: a heading mentioning Courts that follows the England/Wales heading.
            courts_heading = heading.find_next(
                lambda tag: tag.name in {"h3", "h4"}
                and re.search(r"courts?", tag.get_text(" ", strip=True), re.IGNORECASE)
            )
            if courts_heading:
                list_el = courts_heading.find_next("ul")
                if list_el:
                    return list_el

            # Fallback: the first UL after the section heading before the next sibling heading.
            next_section = heading.find_next(
                lambda tag: tag.name in {"h2", "h3", "h4"} and tag is not heading
            )
            cursor = heading
            while True:
                cursor = cursor.find_next(lambda t: t.name in {"ul", "h2", "h3", "h4"})
                if cursor is None or cursor is next_section:
                    break
                if cursor.name == "ul":
                    return cursor
            return None

        courts_list = find_courts_list()
        if not courts_list:
            raise RuntimeError("Could not locate Courts subsection")

        courts: List[Tuple[str, str]] = []
        for li in courts_list.find_all("li", recursive=False):
            link = li.find("a")
            if not link or not link.get("href"):
                continue
            name = link.get_text(strip=True)
            url = urljoin(BASE_URL, link["href"])
            if self.court_filters and not any(
                filt in name.lower() for filt in self.court_filters
            ):
                continue
            courts.append((name, url))
        logging.info("Discovered %d courts", len(courts))
        return courts

    def _extract_year_links(self, soup: BeautifulSoup, court_url: str) -> List[Tuple[str, str]]:
        marker = soup.find(string=lambda s: s and "Or browse the years" in s)
        candidate_container = marker.parent if marker else None
        year_links: List[Tuple[str, str]] = []

        def collect_years(container: Optional[object]) -> None:
            nonlocal year_links
            if not container:
                return
            for link in getattr(container, "find_all", lambda *args, **kwargs: [])(
                "a"
            ):
                text = link.get_text(strip=True)
                if re.fullmatch(r"\d{4}", text):
                    year_links.append((text, urljoin(court_url, link.get("href", ""))))

        collect_years(candidate_container)
        if not year_links and candidate_container is not None:
            collect_years(candidate_container.find_next_sibling())
        if not year_links:
            collect_years(soup)

        return year_links

    def _iter_month_sections(self, soup: BeautifulSoup) -> Iterable[Tuple[str, List[BeautifulSoup]]]:
        headings = []
        for tag in soup.find_all(["h2", "h3", "h4", "h5", "b", "strong"]):
            month = month_from_text(tag.get_text(" ", strip=True))
            if month:
                headings.append((month, tag))

        for idx, (month, tag) in enumerate(headings):
            next_tag = headings[idx + 1][1] if idx + 1 < len(headings) else None
            links: List[BeautifulSoup] = []
            for sibling in tag.next_siblings:
                if sibling is next_tag:
                    break
                if getattr(sibling, "name", None) in {"h2", "h3", "h4", "h5"}:
                    break
                if hasattr(sibling, "find_all"):
                    links.extend(sibling.find_all("a"))
            yield month, links

    def _build_pdf_path(self, court: str, year: str, month: str, title: str) -> Path:
        safe_court = sanitize_for_fs(court)
        safe_year = sanitize_for_fs(str(year))
        safe_month = sanitize_for_fs(month)
        safe_title = sanitize_for_fs(title)
        return self.output_dir / safe_court / safe_year / safe_month / f"{safe_title}.pdf"

    # ------------------------------------------------------------------
    # Processing steps
    # ------------------------------------------------------------------
    async def process_court(self, court_name: str, court_url: str) -> None:
        if self.max_cases_reached:
            return
        logging.info("Processing court: %s", court_name)
        html = await self.fetch_html(court_url)
        soup = BeautifulSoup(html, "html.parser")
        year_links = self._extract_year_links(soup, court_url)
        if not year_links:
            logging.warning("No year links found for %s", court_name)
            return
        for year_text, year_url in year_links:
            if self.max_cases_reached:
                break
            if self.year_range:
                year_int = int(year_text)
                if year_int < self.year_range[0] or year_int > self.year_range[1]:
                    continue
            await self.process_year(court_name, year_text, year_url)

    async def process_year(self, court_name: str, year_text: str, year_url: str) -> None:
        if self.max_cases_reached:
            return
        logging.info("Processing year %s for %s", year_text, court_name)
        html = await self.fetch_html(year_url)
        soup = BeautifulSoup(html, "html.parser")
        for month, link_tags in self._iter_month_sections(soup):
            if self.max_cases_reached:
                break
            await self.process_month(court_name, year_text, month, link_tags)

    async def process_month(
        self, court_name: str, year_text: str, month: str, link_tags: Sequence[BeautifulSoup]
    ) -> None:
        for link in link_tags:
            if self.max_cases_reached:
                break
            href = link.get("href")
            if not href:
                continue
            title = link.get_text(" ", strip=True) or Path(href).stem
            url = urljoin(BASE_URL, href)
            pdf_path = self._build_pdf_path(court_name, year_text, month, title)

            existing = self.progress.get(url)
            pdf_exists = pdf_path.exists()
            status = "downloaded" if pdf_exists else "pending"
            error = None

            if existing:
                status = existing.status
                error = existing.error
                if status == "downloaded" and not pdf_exists:
                    status = "pending"

            record = CaseRecord(
                court=court_name,
                year=year_text,
                month=month,
                title=title,
                url=url,
                pdf_path=str(pdf_path),
                status=status,
                error=error,
            )
            self.case_records.setdefault(record.url, record)

            needs_log = (
                existing is None
                or record.status != existing.status
                or record.error != existing.error
                or record.pdf_path != existing.pdf_path
            )
            if needs_log:
                self.update_record(record)
            else:
                self.progress.setdefault(record.url, record)

            if status == "downloaded":
                logging.debug("Skipping already downloaded case: %s", title)
                continue
            if self.max_cases_reached:
                break

            await self.download_case(record)
            self.processed_cases += 1

    async def download_case(self, record: CaseRecord) -> None:
        await self.ensure_browser()
        pdf_path = Path(record.pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading case: %s", record.title)
        try:
            page: Page = await self.context.new_page()
            await asyncio.sleep(random.uniform(self.delay_min, self.delay_max))
            await page.goto(record.url, wait_until="networkidle", timeout=120000)
            content = await page.content()
            if "excessive traffic" in content.lower():
                logging.warning("Block detected on %s, retrying after backoff", record.url)
                await asyncio.sleep(self.delay_max * 2)
                await page.goto(record.url, wait_until="networkidle", timeout=120000)
            await page.pdf(path=str(pdf_path), print_background=True)
            await page.close()
            record.status = "downloaded"
            record.error = None
            logging.info("Saved PDF to %s", pdf_path)
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error = str(exc)
            logging.exception("Failed to download %s", record.url)
        finally:
            self.update_record(record)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def write_cases_csv(self) -> None:
        records = list(self.progress.values())
        fieldnames = [
            "court",
            "year",
            "month",
            "title",
            "url",
            "pdf_path",
            "status",
            "error",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in sorted(records, key=lambda r: (r.court, r.year, r.month, r.title)):
                writer.writerow({field: getattr(rec, field) for field in fieldnames})
        logging.info("Wrote %d rows to %s", len(records), self.csv_path)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    @property
    def max_cases_reached(self) -> bool:
        return self.max_cases is not None and self.processed_cases >= self.max_cases

    async def run(self) -> None:
        self.load_progress()
        courts = await self.discover_courts()
        try:
            for court_name, court_url in courts:
                if self.max_cases_reached:
                    break
                await self.process_court(court_name, court_url)
        finally:
            await self.close_browser()
            self.write_cases_csv()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polite, resumable BAILII downloader")
    parser.add_argument("--out", required=True, help="Output directory for PDFs and logs")
    parser.add_argument(
        "--courts",
        default="all",
        help="Comma-separated court name filters or 'all'",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Year filter in YYYY or YYYY-YYYY format",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to process (for testing)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=3.0,
        help="Minimum polite delay between requests",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=8.0,
        help="Maximum polite delay between requests",
    )
    parser.add_argument(
        "--resume",
        type=str_to_bool,
        default=True,
        help="Resume from existing progress log",
    )
    parser.add_argument(
        "--headless",
        type=str_to_bool,
        default=True,
        help="Run Playwright in headless mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.delay_min > args.delay_max:
        parser.error("--delay-min cannot be greater than --delay-max")

    downloader = BailiiDownloader(args)
    asyncio.run(downloader.run())


if __name__ == "__main__":
    main()
