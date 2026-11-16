# D2C Qualification Training Dataset

## Overview

This dataset is provided to support algorithm development for the **D2C Qualification Task**.

The Qualification Task evaluates an algorithm's ability to classify the injury state of four extremities on a simulated casualty:

* Left arm
* Right arm
* Left leg
* Right leg

All laterality labels refer to the **casualty's anatomical left and right**, not the left and right sides of the image from the camera or viewer perspective.

The training release contains two types of media:

* Still images
* Video clips

The qualification evaluation set will consist of **images only**. Video clips are provided as supplemental training data.

---

## Dataset Contents

The training package has the following general structure:

```text
dataset/
├── README.md
├── train_dataset.csv
├── images/
│   ├── 0001.jpg/.png
│   ├── 0002.jpg/.png
│   ├── ...
│
└── videos/
    ├── 0003.mp4
    ├── 0004.mp4
    ├── ...
```

`train_dataset.csv` contains the ground-truth labels and file references for all media included in the training release.

### Training Set Size

```text
Total training media:     1190
Images:                   955
Videos:                   235
Unique casualties:        62
```

---

## CSV Schema

The training CSV contains one row per media file.

| Column         | Description                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `casualty_id`  | Identifier associated with a single simulated casualty instance. Multiple media files may share the same `casualty_id`. |
| `filepath`     | Path to the associated image or video file.                                                                             |
| `media_format` | Media type: `image` or `video`.                                                                                         |
| `arm_left`     | Ground-truth injury class for the casualty's anatomical left arm.                                                       |
| `arm_right`    | Ground-truth injury class for the casualty's anatomical right arm.                                                      |
| `leg_left`     | Ground-truth injury class for the casualty's anatomical left leg.                                                       |
| `leg_right`    | Ground-truth injury class for the casualty's anatomical right leg.                                                      |

Example:

```csv
casualty_id,filepath,media_format,arm_left,arm_right,leg_left,leg_right
SP0_T_casualty_6,/Data/dtc_manikin_dataset/images/0001.jpg,image,amputation,no_injury,not_testable,wound
```

Each of the four body-site columns contains exactly one of the following class labels:

```text
no_injury
wound
amputation
not_testable
```

---

## Label Definitions

### `no_injury`

The body site is mostly visible and no qualifying injury is present.

### `wound`

Visible traumatic injury is present, but the limb remains anatomically attached.

Examples may include lacerations, burns, shrapnel wounds, or compound fractures.

### `amputation`

The limb is completely severed.

### `not_testable`

The body site cannot be adequately assessed from the image because it is not sufficiently visible, such as when the limb is outside the image frame or substantially occluded.

The `not_testable` label was assigned **only to image data**. Video data were not annotated for visibility and therefore do not contain `not_testable` labels.

---

## Anatomical Laterality

`left` and `right` always refer to the **casualty's anatomical left and right**.

For example, a casualty's left arm may appear on the right side of an image depending on the camera orientation. Models should therefore determine laterality relative to the casualty rather than the image coordinate system.

---

## Image and Video Labels

For **images**, the four body-site labels describe the injury state that can be determined from that individual image. A body site that cannot be adequately assessed may therefore be labeled `not_testable`.

For **videos**, the labels describe the known injury state of the casualty. Video data were not annotated frame-by-frame for visibility, and `not_testable` was not assigned to video records.

The qualification evaluation uses images only. Teams may use the provided videos as supplemental training data.

---

## Important Notes

* Anatomical laterality is defined relative to the casualty, **not the camera or viewer perspective**.
* `not_testable` is a valid injury class and should not be interpreted as a missing label.
* Only image records were annotated with `not_testable`.
* Multiple media files may correspond to the same `casualty_id`.
* Video data are provided for training only; qualification evaluation inputs will be images.
* A media file may contain injuries to the head, torso, or other anatomical locations. These injuries are **outside the scope of this dataset's labels** and are not represented in the four extremity label columns. Ground-truth labels are provided only for the four specified body sites: left arm, right arm, left leg, and right leg.
* A media file may contain non-PII/PHI visual content in the background that is unrelated to the labeled casualty, such as portions of other manikins or shadows/silhouettes of people. If that is the case, it should be assumed that the casualty being evaluated is the largest manikin in frame.