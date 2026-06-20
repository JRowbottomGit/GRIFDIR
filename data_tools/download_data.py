#!/usr/bin/env python
"""
Download GRIFDIR datasets and pre-built mesh hierarchies from Zenodo into ./data.

Hosted artifacts: Gaussian-blob + multidomain conductivity and pinball flow
sample tensors. (Mesh hierarchies are committed under meshes/.) Training and
evaluation read these directly (no dolfinx required).

    python data_tools/download_data.py

Override the target directory with --dest (default: ./data). Override the
record id with --record or the GRIFDIR_ZENODO_RECORD environment variable.

To regenerate the data from scratch with a fixed seed instead, see the
data_tools/generate_*.py utilities and the README (requires the dolfinx env).
"""
import argparse
import json
import os
import sys
import urllib.request
import zipfile

DEFAULT_RECORD = "20753301"


def fetch_record(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser(description="Download GRIFDIR data from Zenodo.")
    ap.add_argument("--record", default=os.environ.get("GRIFDIR_ZENODO_RECORD", DEFAULT_RECORD),
                    help="Zenodo record id.")
    ap.add_argument("--dest", default="data", help="Target directory (default: ./data).")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    record = fetch_record(args.record)
    files = record.get("files", [])
    print(f"Zenodo record {args.record}: {len(files)} file(s) -> {args.dest}/")
    for f in files:
        name = f["key"]
        out = os.path.join(args.dest, name)
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(f["links"]["self"], out)
        if name.endswith(".zip"):
            print(f"  extracting {name} ...")
            with zipfile.ZipFile(out) as z:
                z.extractall(args.dest)
            os.remove(out)
    print(f"Done -> {args.dest}/")


if __name__ == "__main__":
    main()
