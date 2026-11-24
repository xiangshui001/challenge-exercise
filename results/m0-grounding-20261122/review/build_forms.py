"""Reconstruct a frozen offline review form using verified local or upstream inputs."""
import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "sources.json").read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "challenge-exercise/review-materials"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def store(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = tmp.name
        tmp.write(data)
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def strip_metadata(data):
    """Keep JPEG scans, coding tables, ICC, Adobe and thumbnail-free JFIF headers."""
    clean = bytearray(data[:2])
    pos = 2
    while data[pos:pos + 1] == b"\xff":
        marker = data[pos + 1]
        if marker == 0xda:
            return bytes(clean) + data[pos:]
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if length < 2 or pos + 2 + length > len(data):
            break
        segment = data[pos:pos + 2 + length]
        payload = segment[4:]
        ancillary = 0xe0 <= marker <= 0xef or marker == 0xfe
        keep = ((marker == 0xe2 and payload.startswith(b"ICC_PROFILE\0"))
                or (marker == 0xee and payload.startswith(b"Adobe"))
                or (marker == 0xe0 and payload.startswith(b"JFIF\0") and len(payload) == 14))
        if not ancillary or keep:
            clean.extend(segment)
        pos += 2 + length
    raise ValueError("Invalid JPEG")


def pilot_url(pic):
    args = dict(dataset=SOURCES["whatsup_repository"], config="default", split="Controlled_Images_A",
                offset=pic["row"], length=1)
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(args)
    row = json.loads(download(url))["rows"][0]
    if row["row_idx"] != pic["row"]:
        raise ValueError("Unexpected What'sUp row")
    return row["row"]["image_options"]["src"]


def image_uri(cache, pic, fetch):
    pilot = "row" in pic
    name = f"whatsup/{pic['row']}.jpg" if pilot else pic["image"]
    path = cache / name
    if path.is_file():
        raw = path.read_bytes()
    elif fetch:
        url = pilot_url(pic) if pilot else "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + path.name
        raw = download(url)
    else:
        raise FileNotFoundError(f"Missing {path}. Supply the image or pass --fetch.")
    if digest(raw) == pic["display_sha256"]:
        display = raw
    elif digest(raw) == pic["raw_sha256"]:
        if pilot:
            from PIL import Image, ImageOps
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            image.thumbnail((1100, 1100))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=93)
            display = out.getvalue()
        else:
            display = strip_metadata(raw)
    else:
        raise ValueError(f"Source image mismatch for {name}")
    if digest(display) != pic["display_sha256"]:
        raise ValueError(f"Displayed image mismatch for {name}. For pilot previews use Pillow 12.3.0.")
    if not path.exists():
        store(path, raw)
    return "data:image/jpeg;base64," + base64.b64encode(display).decode("ascii")


def captions(cache, rows, fetch):
    database = {}
    for category in sorted({r["source"].split("/")[1] for r in rows}):
        path = cache / "annotations" / (category + ".json")
        if path.is_file():
            raw = path.read_bytes()
        elif fetch:
            url = f"https://raw.githubusercontent.com/Sri-Harsha/scpp/{SOURCES['scpp_revision']}/data/{category}.json"
            raw = download(url)
        else:
            raise FileNotFoundError(f"Missing {path}. Supply the annotation file or pass --fetch.")
        for r in json.loads(raw):
            database[f"scpp/{category}/{r['id']}"] = (r["caption"], r["caption2"], r["negative_caption"])
        if not path.exists():
            store(path, raw)
    return database


def build(mode, cache, output, fetch):
    spec = SOURCES["forms"][mode]
    annotations = {} if mode == "pair" else captions(cache, spec["rows"], fetch)
    rows = []
    for item in spec["rows"]:
        key = item["id"]
        if mode == "pair":
            images = [image_uri(cache, p, fetch) for p in item["pictures"]]
            row = dict(id=key, imgs=images, texts=item["texts"])
        else:
            lines = [annotations[item["source"]][j] for j in item["permutation"]]
            if [digest(line.encode()) for line in lines] != item["display_text_sha256"]:
                raise ValueError(f"Caption mismatch for {key}")
            image = image_uri(cache, item, fetch)
            if mode == "initial":
                row = dict(id=key, img=image, texts=lines)
            else:
                row = dict(id=key, texts=lines, imgs=[image], priority=item["priority"])
        rows.append(row)
    payload = rows if mode == "initial" else {**spec["header"], "rows": rows}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "templates" / (mode + ".html")).read_text(encoding="utf8")
    html = template.replace("__FORM_DATA__", encoded).replace("\n", "\r\n").encode("utf8")
    if digest(html) != spec["sha256"]:
        raise ValueError("Complete form differs from the frozen offline form")
    path = output / spec["filename"]
    store(path, html)
    print(f"{mode}: verified {len(rows)} entries, wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("form", choices=("initial", "audit", "pair", "all"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download missing public inputs")
    options = parser.parse_args()
    for mode in (SOURCES["forms"] if options.form == "all" else [options.form]):
        build(mode, options.cache, options.out, options.fetch)
