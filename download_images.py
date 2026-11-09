"""
download_images.py
─────────────────────────────────────────────────────────────────────────────
Parallel image downloader for the Amazon product dataset.

Downloads product images from URLs listed in a CSV file and saves them as
<sample_id>.jpg in the output directory. Already-downloaded images are
automatically skipped (idempotent — safe to re-run after interruptions).

USAGE
─────
  # Download all images from the full training set (75K images, ~17 GB):
  python download_images.py --csv dataset/train.csv --outdir data/images --workers 25

  # Download a quick test batch:
  python download_images.py --csv data/sample.csv --outdir data/images --limit 50

ARGUMENTS
─────────
  --csv       Path to CSV with 'sample_id'/'img_id' and 'image_link' columns
  --outdir    Directory to save downloaded images  (default: data/images)
  --workers   Number of parallel download threads  (default: 20)
  --limit     Download only first N images         (optional)

OUTPUT
──────
  data/images/<sample_id>.jpg        — one file per product
  data/images/download_failures.txt  — list of failed URLs (if any)
"""

import os
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -- Configuration --------------------------------------------------------------
DEFAULT_CSV    = "data/dataset_with_images.csv"
DEFAULT_OUTDIR = "data/images"
DEFAULT_WORKERS = 10
TIMEOUT_SEC    = 15
MAX_RETRIES    = 3
RETRY_DELAY    = 2   # seconds between retries
# -------------------------------------------------------------------------------


def download_one(row, out_dir):
    """Download a single image. Returns (filename, success, message)."""
    img_id   = str(row["img_id"])
    url      = str(row["image_link"])
    filename = img_id + ".jpg"
    dest     = out_dir / filename

    # Skip if already downloaded and non-empty
    if dest.exists() and dest.stat().st_size > 500:
        return filename, True, "skipped"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    msg = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT_SEC)
            if resp.status_code == 200 and len(resp.content) > 500:
                dest.write_bytes(resp.content)
                return filename, True, "downloaded"
            else:
                msg = f"HTTP {resp.status_code}, size={len(resp.content)}"
        except requests.exceptions.Timeout:
            msg = "timeout"
        except requests.exceptions.ConnectionError as e:
            msg = f"connection error: {e}"
        except Exception as e:
            msg = f"error: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return filename, False, msg


def main():
    parser = argparse.ArgumentParser(description="Download product images in parallel.")
    parser.add_argument("--csv",     default=DEFAULT_CSV,      help="Path to dataset CSV")
    parser.add_argument("--outdir",  default=DEFAULT_OUTDIR,   help="Directory to save images")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel download threads")
    parser.add_argument("--limit",   type=int, default=None,   help="Limit number of images to download")
    args = parser.parse_args()

    # -- Load data ---------------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Support both 'img_id' (small dataset) and 'sample_id' (full dataset)
    id_col = None
    for candidate in ["img_id", "sample_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    required = {"image_link"}
    if id_col is None or not required.issubset(df.columns):
        print(f"CSV must have an ID column (img_id/sample_id) and 'image_link'. Got: {df.columns.tolist()}")
        return
    df = df.rename(columns={id_col: "img_id"})

    # Drop rows with missing URLs
    before = len(df)
    df = df.dropna(subset=["image_link"])
    df = df[df["image_link"].str.startswith("http")]
    print(f"Loaded {before} rows -> {len(df)} with valid image URLs")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} images")

    # -- Prepare output directory ------------------------------------------------
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {out_dir.resolve()}")

    # -- Check how many already exist -------------------------------------------
    existing     = {f.name for f in out_dir.glob("*.jpg")}
    already_done = sum(1 for _, row in df.iterrows() if f"{row['img_id']}.jpg" in existing)
    to_download  = len(df) - already_done
    print(f"Already downloaded: {already_done}")
    print(f"To download:        {to_download}")

    if to_download == 0:
        print("All images already downloaded! Nothing to do.")
        return

    # -- Download in parallel ---------------------------------------------------
    rows          = [row for _, row in df.iterrows()]
    success_count = 0
    skip_count    = 0
    fail_count    = 0
    failed_rows   = []

    start_time = time.time()
    print(f"\nStarting download with {args.workers} parallel workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, row, out_dir): row for row in rows}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            filename, ok, msg = future.result()

            if msg == "skipped":
                skip_count += 1
            elif ok:
                success_count += 1
            else:
                fail_count += 1
                failed_rows.append((filename, msg))

            # Progress update every 100 items or at the end
            if completed % 100 == 0 or completed == len(rows):
                elapsed = time.time() - start_time
                rate    = completed / max(elapsed, 0.1)
                eta_sec = (len(rows) - completed) / max(rate, 0.01)
                eta_min = eta_sec / 60
                pct     = 100 * completed / len(rows)
                print(
                    f"  [{completed:>6}/{len(rows)}] {pct:5.1f}%  "
                    f"OK:{success_count}  skip:{skip_count}  fail:{fail_count}  "
                    f"| {rate:.1f} img/s  ETA:{eta_min:.1f}min"
                )

    # -- Summary ----------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Newly downloaded : {success_count}")
    print(f"  Already existed  : {skip_count}")
    print(f"  Failed           : {fail_count}")
    print(f"  Total time       : {total_time/60:.1f} minutes")

    if failed_rows:
        fail_log = out_dir / "download_failures.txt"
        with open(fail_log, "w") as f:
            for fname, reason in failed_rows:
                f.write(f"{fname}\t{reason}\n")
        print(f"\nFailed image list saved to: {fail_log}")
        print("Re-run this script to retry them (it will skip successes).")

    total_on_disk = len(list(out_dir.glob("*.jpg")))
    print(f"\nTotal images on disk now: {total_on_disk}")
    print(f"Location: {out_dir.resolve()}")

    if fail_count == 0:
        print("\nAll images downloaded! You can now run training.")
    else:
        coverage = (total_on_disk / len(df)) * 100
        print(f"\nCoverage: {total_on_disk}/{len(df)} images = {coverage:.1f}%")
        print("You can still train with available images (missing ones will be skipped).")


if __name__ == "__main__":
    main()
