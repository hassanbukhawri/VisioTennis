"""
One-time downloader for THETIS's raw RGB video (github.com/THETIS-dataset/dataset),
RGB ONLY -- explicitly not depth/mask/skeleton, per instruction, since only RGB is
needed for video-based fine-tuning (Path B feasibility report, PROJECT_LOG.md
2026-08-12). Uses the GitHub Contents API to list each of the 12 class folders
(12 calls, well under the unauthenticated 60/hour rate limit), then downloads each
file's raw.githubusercontent.com URL directly (does not count against the API
rate limit).

Usage:
    python ir/download_thetis_rgb.py
"""
import json
import os
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/StrokeClassification/raw_rgb")

API_BASE = "https://api.github.com/repos/THETIS-dataset/dataset/contents/VIDEO_RGB"

CLASSES = [
    "backhand", "backhand2hands", "backhand_slice", "backhand_volley",
    "flat_service", "forehand_flat", "forehand_openstands", "forehand_slice",
    "forehand_volley", "kick_service", "slice_service", "smash",
]


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "visiotennis-thetis-downloader"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "visiotennis-thetis-downloader"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    total_files = 0
    total_bytes = 0
    total_skipped = 0
    t_start = time.time()

    for cls in CLASSES:
        cls_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        listing = _get_json(f"{API_BASE}/{cls}")
        print(f"[{cls}] {len(listing)} files listed")

        for entry in listing:
            dest = os.path.join(cls_dir, entry["name"])
            if os.path.exists(dest) and os.path.getsize(dest) == entry["size"]:
                total_skipped += 1
                continue
            _download(entry["download_url"], dest)
            total_files += 1
            total_bytes += entry["size"]
            if total_files % 200 == 0:
                elapsed = time.time() - t_start
                print(f"  ... {total_files} downloaded so far, {total_bytes / (1024**2):.1f} MB, "
                      f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t_start
    print(f"\n[done] downloaded {total_files} new files ({total_bytes / (1024**2):.1f} MB), "
          f"{total_skipped} already present/skipped, in {elapsed:.0f}s")
    print(f"[done] output dir: {OUT_DIR}")
