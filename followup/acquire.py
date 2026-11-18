"""Fetch only registered official COCO/CLEVR files. ZIP member CRC is checked."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import zipfile
import zlib

CLEVR = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
SIZE = 19021600724


def fetch(url, span=None):
    args = ["curl", "-fsSL", "--retry", "4", "--retry-all-errors", "--max-time", "90"]
    if span:
        lo, hi = span
        args += ["--range", f"{lo}-{hi}"]
        url += f"?bytes={lo}-{hi}"
    data = subprocess.check_output(args + [url])
    if span and len(data) != hi-lo+1:
        raise ValueError("Incorrect byte range")
    return data


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0

    def seek(self, pos, whence=0):
        self.pos = (0, self.pos, SIZE)[whence]+pos
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        end = SIZE if size < 0 else min(SIZE, self.pos+size)
        if end <= self.pos:
            return b""
        data = fetch(CLEVR, (self.pos, end-1))
        self.pos = end
        return data


def member(info):
    start = info.header_offset
    buf = fetch(CLEVR, (start, start+info.compress_size+1023))
    assert buf[:4] == b"PK\x03\x04"
    a, b = struct.unpack_from("<HH", buf, 26)
    offset = 30+a+b
    data = buf[offset:offset+info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    assert len(data) == info.file_size and zlib.crc32(data) == info.CRC
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["scenes", "clevr", "coco"])
    p.add_argument("out", type=Path)
    p.add_argument("--manifest", type=Path)
    a = p.parse_args()
    if a.mode != "coco":
        archive = zipfile.ZipFile(RangeFile())
    if a.mode == "scenes":
        a.out.write_bytes(member(archive.getinfo("CLEVR_v1.0/scenes/CLEVR_val_scenes.json")))
        return
    names = [json.loads(l)["images"][0] for l in a.manifest.read_text(encoding="utf8").splitlines()]
    def download(name):
        dest = a.out/name
        if dest.exists():
            return
        if a.mode == "clevr":
            data = member(archive.getinfo("CLEVR_v1.0/images/val/"+dest.name))
        else:
            data = fetch("https://s3.amazonaws.com/images.cocodataset.org/val2017/"+dest.name)
            if not data.startswith(b"\xff\xd8"):
                raise ValueError(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".part")
        temp.write_bytes(data)
        os.replace(temp, dest)
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(download, (n for n in names if n.startswith(a.mode+"/"))))


if __name__ == "__main__":
    main()

