#!/usr/bin/env python3
"""Convert the local FiftyOne CropAndWeed export to two-class YOLO detection.

Uses the authors' CropOrWeed2 mapping; unmapped species/Vegetation are omitted.
Missing ground truth, invalid boxes, and images without retained boxes are skipped.
Splits are deterministic 70/10/20 by recording-set/session (not image count).
Images are hard-linked by default; use --copy-images for independent copies.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def yolo_box(box, width=1, height=1):
    """Convert top-left xywh to normalized center xywh; reject invalid boxes."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h, width, height)):
        return None
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    if x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        return None
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def export(records, output, stats, policy, copy_images=False):
    """Create a fresh output; hard-link images where possible, otherwise copy."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Use --output with a new directory.")
    # Preflight before writing anything.
    names = set()
    for src, name, split, lines in records:
        if not src.is_file():
            raise FileNotFoundError(src)
        if name in names:
            raise ValueError(f"Duplicate output image name: {name}")
        names.add(name)
    if not records:
        raise ValueError("No usable annotated images found")
    (output / 'images').mkdir(parents=True)
    (output / 'labels').mkdir()
    splits = {key: [] for key in ('train', 'val', 'test')}
    for src, name, split, lines in records:
        dest = output / 'images' / name
        if copy_images:
            shutil.copy2(src, dest)
        else:
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
        (output / 'labels' / Path(name).with_suffix('.txt')).write_text('\n'.join(lines) + '\n')
        splits[split].append('./images/' + name)
        stats['images_exported'] += 1
        stats['boxes_exported'] += len(lines)
        for line in lines:
            stats['crop_boxes' if line.startswith('0 ') else 'weed_boxes'] += 1
    for split, paths in splits.items():
        (output / f'{split}.txt').write_text('\n'.join(paths) + ('\n' if paths else ''))
        stats[f'{split}_images'] = len(paths)
    (output / 'data.yaml').write_text(
        f'path: {json.dumps(str(output.resolve()))}\n'
        'train: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: crop\n  1: weed\n'
    )
    report = {'counts': dict(stats), 'policy': policy}
    (output / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def label_line(cls, box):
    return str(cls) + ' ' + ' '.join(f'{v:.10f}' for v in box)


import random

# Source: https://github.com/cropandweed/cropandweed-dataset/blob/main/cnw/utilities/datasets.py
CROP_IDS = set(range(1, 13)) | {94, 24, 18, 13, 26, 27, 15}
WEED_IDS = {31,48,62,65,68,69,74,75,81,84,86,32,29,33,37,49,30,44,66,87,89,91,
            61,79,34,41,52,35,36,78,38,39,71,72,88,42,45,70,47,51,54,58,60,80,
            83,96,22,63,85,56,57,64,77,50,59,67,76}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/CropAndWeed')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/YOLO_DATA/CAW')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args()
    samples = json.loads((args.source / 'samples.json').read_text())['samples']
    stats = Counter(images_input=len(samples))
    pending = []
    skipped_ids = Counter()
    for sample in samples:
        gt = sample.get('ground_truth')
        if not isinstance(gt, dict) or 'detections' not in gt:
            stats['images_missing_ground_truth'] += 1
            continue
        lines = []
        for detection in gt['detections'] or []:
            stats['boxes_input'] += 1
            box = yolo_box(detection['bounding_box'])
            if box is None:
                stats['boxes_invalid'] += 1
                continue
            label_id = int(detection['label_id'])
            if label_id not in CROP_IDS | WEED_IDS:
                stats['boxes_unmapped'] += 1
                skipped_ids[label_id] += 1
                continue
            lines.append(label_line(0 if label_id in CROP_IDS else 1, box))
        if not lines:
            stats['images_no_usable_boxes'] += 1
            continue
        src = args.source / sample['filepath']
        group = (sample['recording_set'], sample['session'])
        pending.append((src, src.name, group, lines))
    groups = sorted({record[2] for record in pending})
    if len(groups) < 3:
        raise ValueError('At least three recording sessions are required for train/val/test')
    random.Random(args.seed).shuffle(groups)
    ntrain = min(len(groups) - 2, max(1, int(len(groups) * .7)))
    nval = max(1, int(len(groups) * .1))
    assignments = {group: ('train' if i < ntrain else 'val' if i < ntrain + nval else 'test')
                   for i, group in enumerate(groups)}
    records = [(src, name, assignments[group], lines) for src, name, group, lines in pending]
    policy = {'class_mapping': 'Official CropOrWeed2', 'crop_source_ids': sorted(CROP_IDS),
              'weed_source_ids': sorted(WEED_IDS), 'unmapped_box_counts_by_source_id': dict(skipped_ids),
              'empty_images': 'excluded', 'split': '70/10/20 recording-set/session groups',
              'seed': args.seed, 'group_assignments': {'/'.join(k): v for k, v in assignments.items()},
              'images': 'copy' if args.copy_images else 'hardlink with copy fallback'}
    export(records, args.output, stats, policy, args.copy_images)


if __name__ == '__main__':
    main()
