"""Validate the small, versioned in-row image-label pilot (standard library only)."""

import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "pilot_data" / "row_decisions"
EXPECTED = Counter(
    {"OPEN": 51, "WEEDS_CONTINUE": 6, "CROP_STOP": 4, "OVERGROWTH": 6}
)
REQUIRED = {
    "image",
    "label",
    "crop",
    "recording",
    "timestamp_ns",
    "reviewer",
    "review_note",
    "sha256",
}


def main() -> None:
    with (ROOT / "labels.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED.issubset(reader.fieldnames or []):
            missing = sorted(REQUIRED - set(reader.fieldnames or []))
            raise ValueError(f"Missing columns: {missing}")
        records = list(reader)

    seen = set()
    counts = Counter()
    for record in records:
        relative = Path(record["image"])
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "images"
            or ".." in relative.parts
        ):
            raise ValueError(f"Invalid image path: {relative}")
        if relative in seen:
            raise ValueError(f"Duplicate image: {relative}")
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        if path.stem.rsplit("_", 1)[-1] != record["timestamp_ns"]:
            raise ValueError(f"Timestamp mismatch: {relative}")
        if not record["crop"] or not record["recording"] or not record["reviewer"]:
            raise ValueError(f"Missing provenance: {relative}")
        counts[record["label"]] += 1

    on_disk = {
        path.relative_to(ROOT) for path in (ROOT / "images").rglob("*.jpg")
    }
    if on_disk != seen:
        raise ValueError(
            f"Image/manifest mismatch: extra={on_disk - seen}, missing={seen - on_disk}"
        )
    if counts != EXPECTED:
        raise ValueError(f"Unexpected label counts: {counts} != {EXPECTED}")
    print(f"Validated {len(records)} frames: {dict(counts)}")


if __name__ == "__main__":
    main()
