#!/usr/bin/env python3
"""Convert ACRE COCO boxes to two-class YOLO detection, preserving official splits.

The supplied COCO file already maps XML 'unknow' plants to weed. Invalid boxes
and images without boxes are excluded. No segmentation or stem data is exported.
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


from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/The_ACRE_Crop-Weed_Dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/YOLO_DATA/ACRE')
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args()
    data = json.loads((args.source / 'data/ACRE_COCO_annotations.json').read_text())
    split_data = json.loads((args.source / 'split_dictionary.json').read_text())
    assignments = {}
    for source_split, target in [('train', 'train'), ('test_dev', 'val'), ('test_final', 'test')]:
        for name in split_data[source_split]:
            if name in assignments:
                raise ValueError(f'Image occurs in multiple splits: {name}')
            assignments[name] = target
    category_map = {c['id']: {'crop': 0, 'weed': 1}[c['name']] for c in data['categories']}
    by_image = defaultdict(list)
    image_ids = {i['id'] for i in data['images']}
    for annotation in data['annotations']:
        if annotation['image_id'] not in image_ids:
            raise ValueError('Annotation references an unknown image')
        by_image[annotation['image_id']].append(annotation)
    stats = Counter(images_input=len(data['images']))
    records = []
    for image in data['images']:
        lines = []
        annotations = by_image[image['id']]
        if not annotations:
            stats['images_without_annotations'] += 1
            continue
        for annotation in annotations:
            stats['boxes_input'] += 1
            box = yolo_box(annotation['bbox'], image['width'], image['height'])
            if box is None:
                stats['boxes_invalid'] += 1
                continue
            lines.append(label_line(category_map[annotation['category_id']], box))
        if not lines:
            stats['images_no_usable_boxes'] += 1
            continue
        src = args.source / 'data' / image['file_name']
        records.append((src, src.name, assignments[src.name], lines))
    export(records, args.output, stats,
           {'class_mapping': 'COCO crop -> 0, weed -> 1; source unknown plants already folded into weed',
            'split': 'official train/test_dev/test_final -> train/val/test',
            'empty_images': 'excluded',
            'images': 'copy' if args.copy_images else 'hardlink with copy fallback'}, args.copy_images)


if __name__ == '__main__':
    main()
