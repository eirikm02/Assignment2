"""Download only the files listed in out/download_list.txt from the remote OSSGameBench.zip.
Usage:  python download_commits.py "<dropbox url with dl=1>" out/download_list.txt [more paths...]
Files are written under ./ (so ./OSSGameBench/commits/...). Already-downloaded files are skipped."""
import os, sys
from remotezip import RemoteZip
url, lst = sys.argv[1], sys.argv[2]
names = [l.strip() for l in open(lst, encoding="utf-8") if l.strip()] + sys.argv[3:]
with RemoteZip(url) as z:
    for i, n in enumerate(names, 1):
        if os.path.exists(n):
            continue
        z.extract(n, ".")
        print(f"[{i}/{len(names)}] {n}", flush=True)
