# In-row decision pilot

This pilot complements—not replaces—the repository's existing two-class YOLO
`crop`/`weed` detector. Object identity alone does not say whether the SPV can
follow a row without hitting a crop. We therefore sampled **forward, low-angle
camera views from a ground robot in crop rows** and reviewed the apparent travel
lane at the **whole-image** level. We did not collect isolated plant portraits
or invent YOLO bounding boxes for these frames.

## What's here

- `images/`: 73 original 832 × 468 rectified left-camera JPEGs from five
  TerraSentia ROS bag recordings. Only collaborator-approved frames are present.
- `labels.csv`: one row per image, with the image-level label, crop, source
  recording, ROS timestamp (nanoseconds), review note, and SHA-256 checksum.
- `python scripts/validate_row_decisions.py`: checks labels, image paths,
  checksums, and counts from the repository root.

| Label | Count | Review meaning |
| --- | ---: | --- |
| `OPEN` | 57 | Visible row judged clear to continue. |
| `WEEDS_CONTINUE` | 6 | Vegetation judged to be weeds in the path; reviewer would continue. |
| `CROP_STOP` | 4 | Apparent crop-in-path risk; reviewer would stop. |
| `OVERGROWTH` | 6 | Dense/overhanging growth; the motion response is **undecided**. |

These are the collaborator's visual judgments, not verified botanical species
labels, measured vehicle clearances, or live motion commands. In particular,
`OVERGROWTH` is a **perception condition**, not a synonym for `SLOW` or `STOP`.
A future controller would need a separate policy using path visibility,
vehicle footprint, obstacle sensing, and uncertainty. People and non-plant
obstacles are not represented here; a separate independent stop mechanism is
required before any moving-robot test.

## Why this direction

The current detector can learn to put boxes around crops and weeds. It may be
useful as one input to navigation, but a box does not establish whether a plant
is inside the SPV's wheel/screw footprint, whether the row ends, or whether a
camera is blocked by leaves. We started with the decision the vehicle actually
needs to make—whether the **visible corridor** appears usable—and kept the
ambiguous overgrowth cases distinct. The two approaches can later be combined:
crop/weed detections, row geometry, and a distance/stop sensor can inform a
conservative motion policy.

## How the pilot was assembled

1. Selected TerraSentia recordings explicitly marked `one_row`: early corn
   (June 13 and 20, 2022), sweet corn (July 12), and soybean (July 12 and 25).
2. Extracted the ROS topic
   `/terrasentia/zed2/zed_node/left/image_rect_color/compressed` without
   upscaling. Sampled frames about 8–10 seconds apart; searched the sweet-corn
   run more densely for difficult examples.
3. Reviewed numbered contact sheets collaboratively, discarding out-of-row,
   redundant, or unwanted frames. An earlier sweet-corn review added 18 accepted
   frames from the denser scan. A subsequent review added six `OPEN` frames
   from the two cornfield runs and sweet corn. Frames explicitly marked for
   deletion were excluded. The separate, unreviewed sweet-corn sheet was not included.
4. Copied only accepted images and labels here. The original large `.bag` and
   `.svo` recordings, review sheets, and rejected frames are **not** included.

There is no train/validation/test split here. Many frames from the same video
are highly correlated, and the set is small and imbalanced (57/73 `OPEN`). A
random frame split would overstate generalization. If this proceeds beyond a
hackathon demo, evaluate on separate recordings, days, fields, crop stages, and
the actual SPV camera viewpoint. The TerraSentia camera's height and alignment
have not been verified against the proposed SPV mount. **No model was trained
or deployed from this pilot.**

## Source and license

Source: [TerraSentia under-canopy dataset](https://github.com/jrcuaranv/terrasentia-dataset)
by Cuaran et al., described in *Under-canopy dataset for advancing simultaneous
localization and mapping in agricultural robotics*, *The International Journal
of Robotics Research* 43(6), 739–749 (2023). The source recordings are linked
from the authors' [cornfield1](https://github.com/jrcuaranv/terrasentia-dataset/blob/main/download_scripts/cornfield1_links.sh),
[sweet-corn](https://github.com/jrcuaranv/terrasentia-dataset/blob/main/download_scripts/sweet_corn_links.sh),
and [soybean](https://github.com/jrcuaranv/terrasentia-dataset/blob/main/download_scripts/soybean_links.sh)
download scripts. The source dataset is licensed [CC BY-SA 4.0](https://github.com/jrcuaranv/terrasentia-dataset/blob/main/license.txt).
These frames were extracted and selected from the recordings; the image-level
labels and review notes were added for this pilot. The contents of this
`pilot_data/row_decisions/` directory are shared under **CC BY-SA 4.0** with
this attribution and change notice. This statement does not change the license
of unrelated repository code or training artifacts.
