"""
download_dataset.py
===================
Fetches the real news corpus used for the results reported in the paper.

The corpus is a public, widely used collection of ~6,300 labelled news articles
(REAL / FAKE).  It is not bundled with this submission because the raw CSV is
about 30 MB, which would push the archive well past the upload limit.  Running
this script once restores the exact file the pipeline was evaluated on.

Usage
-----
    python download_dataset.py
    python main.py

If you have no network access, skip this step entirely.  `main.py` falls back to
`data/news_dataset.csv`, the synthetic corpus that ships with the project, and
every stage of the pipeline runs unchanged - only the numbers differ, and the
README explains why.

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/"
       "fake_or_real_news.csv")
OUT = os.path.join("data", "train.csv")

# Row counts after de-duplication, as used in the report.  Printed so you can
# confirm you have the same corpus rather than a silently different revision.
EXPECTED_ROWS = 6060
EXPECTED_REAL = 2989
EXPECTED_FAKE = 3071


def main() -> int:
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUT):
        print(f"{OUT} already exists - nothing to do.")
        return 0

    print(f"Downloading from {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=120) as r:
            raw = r.read()
    except Exception as exc:                       # network blocked, offline, etc.
        print(f"\nDownload failed: {exc}")
        print("No problem - run `python main.py` and it will use the bundled")
        print("synthetic corpus at data/news_dataset.csv instead.")
        return 1

    print(f"  received {len(raw) / 1e6:.1f} MB "
          f"(sha256 {hashlib.sha256(raw).hexdigest()[:16]}...)")

    try:
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(raw))
        df = (df.dropna(subset=["text", "label"])
                .drop_duplicates(subset=["text"])
                .reset_index(drop=True))
        # The published file labels rows with the strings REAL / FAKE.
        df["label"] = (df["label"].astype(str).str.upper() == "FAKE").astype(int)
        df = df[["id", "title", "text", "label"]]
        df.to_csv(OUT, index=False)

        n, real, fake = len(df), int((df.label == 0).sum()), int((df.label == 1).sum())
        print(f"\nWrote {OUT}")
        print(f"  articles   : {n}      (expected {EXPECTED_ROWS})")
        print(f"  REAL (0)   : {real}   (expected {EXPECTED_REAL})")
        print(f"  FAKE (1)   : {fake}   (expected {EXPECTED_FAKE})")

        if (n, real, fake) != (EXPECTED_ROWS, EXPECTED_REAL, EXPECTED_FAKE):
            print("\nNote: these counts differ from the ones behind the reported")
            print("results, so your numbers will not match the report exactly.")
            print("The upstream file has probably been revised since.")
        else:
            print("\nMatches the corpus used in the report. Run `python main.py`.")
        return 0

    except ImportError:
        with open(OUT, "wb") as f:
            f.write(raw)
        print(f"\nWrote raw file to {OUT} (pandas unavailable, so labels were not")
        print("converted - main.py will handle the REAL/FAKE strings itself).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
