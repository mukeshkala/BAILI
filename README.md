# BAILII Downloader

A polite, resumable downloader for England and Wales case law courts on [BAILII](https://www.bailii.org/databases.html). It crawls the court listings, walks year and month archives, and saves each case page as a PDF using Playwright.

## Features

- Discovers courts automatically from the **England and Wales Case Law → Courts** section.
- Follows “Or browse the years …” links per court and collects cases by month.
- Generates PDFs via Playwright (Chromium) with background graphics enabled.
- Polite scraping defaults: single concurrency, random delays, retries with backoff, block detection.
- Resume support through an append-only `progress.jsonl` log and `cases.csv` metadata export.
- Windows-safe folder/file names: `<OUT>/<Court>/<Year>/<Month>/<Case>.pdf`.

## Setup

```bash
pip install -r requirements.txt
python -m playwright install
```

## Usage

```bash
python bailii_downloader.py --out ./data --courts all
```

Common options:

- `--courts all|<comma-separated filters>`: limit courts by name substring.
- `--years <from>-<to>`: year filter (e.g., `2019-2020` or `2021`).
- `--year-list <y1,y2,...>`: explicitly list years (comma or space separated).
- `--max-cases <n>`: stop after N case downloads (useful for smoke tests).
- `--court-url <url>`: skip discovery and start from a specific court listing URL.
- `--court-name <name>`: friendly court name to use when `--court-url` is set.
- `--delay-min/--delay-max`: random polite delay bounds (seconds).
- `--resume true|false`: reuse `progress.jsonl` to skip finished cases.
- `--headless true|false`: toggle Playwright headless mode.
- `--log-level INFO|DEBUG|WARNING|ERROR`.

### Example commands

Download a small sample for testing:

```bash
python bailii_downloader.py --out ./data --courts "Court of Appeal" --max-cases 5 --log-level DEBUG
```

Resume after an interrupted run (default behavior):

```bash
python bailii_downloader.py --out ./data --courts all --resume true
```

Run Playwright with a visible browser:

```bash
python bailii_downloader.py --out ./data --courts all --headless false
```

Use an explicit court and year list (e.g., EWCA Civ years provided manually):

```bash
python bailii_downloader.py \
  --out ./data \
  --court-url https://www.bailii.org/ew/cases/EWCA/Civ/ \
  --year-list "2019 2020 2021" \
  --court-name "Court of Appeal (Civil)"
```

## Resume behavior

- `progress.jsonl` keeps a per-case status (`pending|downloaded|failed`).
- On startup, the downloader loads the log (if `--resume true`) and skips cases whose PDFs already exist.
- `cases.csv` is regenerated each run to summarize the latest statuses and file paths.

## Notes

- Default concurrency is 1 with 3–8s randomized delays to reduce load on BAILII.
- The downloader checks for “excessive traffic” messages and backs off automatically.
- Filenames and folder names are sanitized to avoid Windows path issues.
