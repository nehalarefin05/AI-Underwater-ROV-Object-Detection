# DATASET AUDIT

**Project:** AI-Based Underwater Visual Perception for Real-Time ROV Object Detection  
**Stage:** Dataset inspection / audit only - no training, no conversion, no modification  
**Audit date:** 2026-10-01  
**Workspace:** `C:\Nehal uiu\AI Vission`  
**Dataset integrity guarantee:** no file in the dataset was created, modified, renamed, moved, converted or deleted. All analysis was read-only.

---

## 1. Dataset overview

| Field | Value |
|---|---|
| Dataset | **Seaclear Marine Debris Dataset** |
| Domain | Underwater marine litter + observed animals + ROV parts, ROV-captured |
| Total images | **8,610** |
| Total annotations | **31,555** |
| Number of classes | **40** |
|-- dataset.json                (37.8 MB - COCO: categories/images/annotations)
|
| Annotation format | **COCO 1.0** (detection **+** instance segmentation polygons) |
| Already YOLO format? | **No** |
| Subsets present | **11** (`<Site>/<Camera>`) |
| Subsets merged? | **No - kept separate as instructed** |
| Existing train/val/test split | **None** |
| Image resolution | 1920x1080 (100% of images) |
| Corrupted images | **0** |
| Ready for YOLO training? | **No** - conversion + split + yaml required |

Sites: Bistrina, Jakljan, Lokrum, Marseille, Slano  
Cameras: `Bluerobotics HD`, `Paralenz Vaquita`, `Paralenz Vaquita Gen 2`, `SIP-E323CV`

---

## 2. Folder structure

```
AI Vission\
|-- README.txt                   (678 B - upstream description)
|-- dataset.json                 (37.8 MB - COCO: categories / images / annotations)
|
|-- Bistrina/                    (site, Croatia)
|   |-- Bluerobotics HD/         1390 imgs   6986 inst   28/40 classes
|   |-- Paralenz Vaquita Gen 2/  2069 imgs  13527 inst   30/40 classes
|   `-- SIP-E323CV/               193 imgs    751 inst   18/40 classes
|-- Jakljan/                     (site, Croatia)
|   |-- Bluerobotics HD/          241 imgs    793 inst   18/40 classes
|   `-- Paralenz Vaquita/          65 imgs    164 inst    7/40 classes
|-- Lokrum/                      (site, Croatia)
|   |-- Bluerobotics HD/          556 imgs   2480 inst   24/40 classes
|   |-- Paralenz Vaquita Gen 2/    77 imgs    913 inst   13/40 classes
|   `-- SIP-E323CV/               339 imgs    691 inst   13/40 classes
|-- Marseille/                   (site, France)
|   `-- SIP-E323CV/              3441 imgs   4094 inst    5/40 classes
`-- Slano/                       (site, Croatia)
    |-- Bluerobotics HD/          168 imgs    932 inst   23/40 classes
    `-- Paralenz Vaquita/          71 imgs    224 inst    7/40 classes
```

Structural notes:
- Depth is exactly `<Site>/<Camera>/<file>.jpg` for all 8,610 images - no stray nesting,
  no `images/` / `labels/` pair, no split directories.
- Every subset is a **flat** dump of `*.jpg`, so bare filenames restart per folder
  (e.g. `1.jpg` exists in several subsets). The COCO `file_name` values are nevertheless
  globally unique and map 1:1 onto disk, so the JSON joins cleanly - but a YOLO export must
  flatten or prefix filenames so the output stays unique.
- There is exactly **one** annotation file for the entire dataset: `dataset.json`.
  There are no per-image `.txt` labels and no `dataset.yaml`.

### 2.1 Subset sizes

| Subset (Site / Camera) | Images | Share | Instances | Inst/img | Classes present |
|---|---:|---:|---:|---:|---:|
| Bistrina / Bluerobotics HD | 1390 | 16.1% | 6986 | 5.03 | 28/40 |
| Bistrina / Paralenz Vaquita Gen 2 | 2069 | 24.0% | 13527 | 6.54 | 30/40 |
| Bistrina / SIP-E323CV | 193 | 2.2% | 751 | 3.89 | 18/40 |
| Jakljan / Bluerobotics HD | 241 | 2.8% | 793 | 3.29 | 18/40 |
| Jakljan / Paralenz Vaquita | 65 | 0.8% | 164 | 2.52 | 7/40 |
| Lokrum / Bluerobotics HD | 556 | 6.5% | 2480 | 4.46 | 24/40 |
| Lokrum / Paralenz Vaquita Gen 2 | 77 | 0.9% | 913 | 11.86 | 13/40 |
| Lokrum / SIP-E323CV | 339 | 3.9% | 691 | 2.04 | 13/40 |
| Marseille / SIP-E323CV | 3441 | 40.0% | 4094 | 1.19 | 5/40 |
| Slano / Bluerobotics HD | 168 | 2.0% | 932 | 5.55 | 23/40 |
| Slano / Paralenz Vaquita | 71 | 0.8% | 224 | 3.15 | 7/40 |
| **TOTAL** | **8610** | **100%** | **31555** | **3.66** | **40/40** |

| Axis | Distribution |
|---|---|
| By site | Bistrina: 3652, Jakljan: 306, Lokrum: 972, Marseille: 3441, Slano: 239 |
| By camera | Bluerobotics HD: 2355, Paralenz Vaquita: 136, Paralenz Vaquita Gen 2: 2146, SIP-E323CV: 3973 |

---

## 3. Dataset statistics

| Metric | Value |
|---|---:|
| Files in workspace | 8,612 (8,610 `.jpg` + `dataset.json` + `README.txt`) |
| Image entries in `dataset.json` | 8,610 |
| Annotation records | 31,555 |
| Categories declared | 40 |
| Images with >= 1 annotation | **8,610 (100%)** |
| Images with 0 annotations | **0** |
| Mean objects per image | 3.67 |
| Median objects per image | 2 |
| Min / max objects per image | 1 / 74 |
| Images with exactly 1 object | 3,628 (42.1%) |
| Images with >= 5 objects | 2,423 (28.1%) |
| Total size on disk | 1.75 GB |
| Image file size min / median / max | 68 KB / 152 KB / 1.01 MB |

### 3.1 Image formats and resolution

| Property | Value |
|---|---|
| Formats | JPEG - 8,610 (100%) |
| Colour mode | RGB - 8,610 (100%) |
| Resolution | **1920x1080 - 8,610 (100%)** |
| Distinct resolutions | 1 |
| Aspect ratio | 16:9 (1.778) - 8,610 (100%) |

The perfectly uniform 1920x1080 / 16:9 profile is a genuine advantage: no padding, no letterboxing, and a single letterbox value for the whole dataset.

---

## 4. Annotation format

```json
{
  "categories":  [ 40 entries ],
  "images":      [ 8610 entries: { "id", "file_name", "width", "height" } ],
  "annotations": [ 31555 entries ]
}
```

```json
// one annotation record
{ "id": 0, "image_id": 0, "category_id": 1,
  "bbox": [983, 673, 50, 101],        // [x, y, width, height] in ABSOLUTE PIXELS
  "area": 5050,                        // == w * h
  "segmentation": [[985,682, 982,759, 997,774, 1021,762, 1032,688, 1012,673]],
  "iscrowd": 0 }
```

| Property | Finding |
|---|---|
| Format | **COCO 1.0** (not YOLO) |
| bbox layout | `[x_min, y_min, width, height]` |
| bbox units | **Absolute pixels** (max coordinate observed = 1920) |
| YOLO normalised `cx,cy,w,h`? | **No** - conversion required |
| segmentation | Present on **31,555 / 31,555** (100%), polygon format |
| iscrowd | `0` on 100% of records - no crowd regions to handle |
| `area` field | Consistent with `w*h` on all records |
| Optional image fields | `license`, `url`, `date_captured` present on only the 3,441 Marseille entries |
| Per-image `.txt` labels | **None** |
| `dataset.yaml` | **None** |

**Note:** because every annotation also carries a polygon `segmentation`, the same source data can later support a YOLO *segmentation* variant, not just detection.

---

## 5. Class names and IDs

**40 classes.** COCO category IDs run **1..40** (1-based). Ultralytics YOLO requires contiguous **0-based** IDs, so the export must apply:

```
yolo_class_id = coco_category_id - 1      # 1..40  ->  0..39
```

The class list below is in COCO ID order, which is exactly the order required in `names:` inside `dataset.yaml`.

| COCO ID | YOLO ID | Class name | Instances | Images | Subsets | Median box side (px) |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | `can_metal` | 1130 | 811 | 9/11 | 72.0 |
| 2 | 1 | `tarp_plastic` | 31 | 31 | 3/11 | 233.5 |
| 3 | 2 | `container_plastic` | 84 | 84 | 3/11 | 167.9 |
| 4 | 3 | `bottle_plastic` | 1261 | 1127 | 10/11 | 105.3 |
| 5 | 4 | `tube_cement` | 1404 | 1139 | 6/11 | 426.3 |
| 6 | 5 | `plant` | 472 | 418 | 6/11 | 150.7 |
| 7 | 6 | `container_middle_size_metal` | 60 | 60 | 3/11 | 189.4 |
| 8 | 7 | `animal_etc` | 3749 | 2007 | 6/11 | 134.4 |
| 9 | 8 | `animal_sponge` | 1110 | 731 | 4/11 | 194.1 |
| 10 | 9 | `bottle_glass` | 2331 | 1355 | 8/11 | 140.0 |
| 11 | 10 | `wreckage_metal` | 265 | 171 | 4/11 | 189.8 |
| 12 | 11 | `unknown_instance` | 1195 | 884 | 8/11 | 165.1 |
| 13 | 12 | `pipe_plastic` | 155 | 154 | 6/11 | 148.0 |
| 14 | 13 | `net_plastic` | 960 | 900 | 3/11 | 660.9 |
| 15 | 14 | `animal_shells` | 995 | 523 | 5/11 | 96.7 |
| 16 | 15 | `rope_fiber` | 2039 | 1356 | 7/11 | 370.0 |
| 17 | 16 | `animal_urchin` | 6462 | 1999 | 6/11 | 59.5 |
| 18 | 17 | `cup_plastic` | 329 | 322 | 7/11 | 79.7 |
| 19 | 18 | `brick_clay` | 606 | 416 | 4/11 | 168.8 |
| 20 | 19 | `bag_plastic` | 882 | 852 | 7/11 | 89.9 |
| 21 | 20 | `sanitaries_plastic` | 55 | 55 | 4/11 | 95.4 |
| 22 | 21 | `clothing_fiber` | 298 | 291 | 9/11 | 251.4 |
| 23 | 22 | `cup_ceramic` | 123 | 113 | 6/11 | 40.4 |
| 24 | 23 | `boot_rubber` | 161 | 142 | 4/11 | 272.6 |
| 25 | 24 | `tire_rubber` | 2556 | 2260 | 8/11 | 331.0 |
| 26 | 25 | `jar_glass` | 62 | 50 | 3/11 | 169.2 |
| 27 | 26 | `rov_cable` | 389 | 362 | 8/11 | 624.0 |
| 28 | 27 | `rov_tortuga` | 55 | 55 | 3/11 | 56.3 |
| 29 | 28 | `branch_wood` | 430 | 203 | 4/11 | 206.7 |
| 30 | 29 | `furniture_wood` | 15 | 12 | 1/11 | 108.0 |
| 31 | 30 | `snack_wrapper_plastic` | 172 | 172 | 3/11 | 147.9 |
| 32 | 31 | `lid_plastic` | 20 | 20 | 2/11 | 60.3 |
| 33 | 32 | `cardboard_paper` | 13 | 13 | 1/11 | 125.3 |
| 34 | 33 | `rope_plastic` | 8 | 8 | 1/11 | 634.6 |
| 35 | 34 | `cable_metal` | 149 | 143 | 1/11 | 453.3 |
| 36 | 35 | `animal_fish` | 985 | 81 | 6/11 | 34.0 |
| 37 | 36 | `snack_wrapper_paper` | 8 | 6 | 2/11 | 58.7 |
| 38 | 37 | `rov_vehicle_leg` | 461 | 461 | 1/11 | 257.7 |
| 39 | 38 | `rov_bluerov` | 58 | 58 | 3/11 | 220.1 |
| 40 | 39 | `animal_starfish` | 17 | 17 | 1/11 | 64.8 |

> Every one of the 40 classes has at least one instance - there are **no empty classes**.

---

## 6. Class distribution

**31555 instances across 40 classes.** Mean 788.88 instances/class, median 313.5.

| Rank | Class | Instances | Cumulative % |
|---:|---|---:|---:|
| 1 | `animal_urchin` | 6462 | 20.48% |
| 2 | `animal_etc` | 3749 | 32.36% |
| 3 | `tire_rubber` | 2556 | 40.46% |
| 4 | `bottle_glass` | 2331 | 47.85% |
| 5 | `rope_fiber` | 2039 | 54.31% |
| 6 | `tube_cement` | 1404 | 58.76% |
| 7 | `bottle_plastic` | 1261 | 62.75% |
| 8 | `unknown_instance` | 1195 | 66.54% |
| 9 | `can_metal` | 1130 | 70.12% |
| 10 | `animal_sponge` | 1110 | 73.64% |
| 11 | `animal_shells` | 995 | 76.79% |
| 12 | `animal_fish` | 985 | 79.91% |
| 13 | `net_plastic` | 960 | 82.96% |
| 14 | `bag_plastic` | 882 | 85.75% |
| 15 | `brick_clay` | 606 | 87.67% |
| 16 | `plant` | 472 | 89.17% |
| 17 | `rov_vehicle_leg` | 461 | 90.63% |
| 18 | `branch_wood` | 430 | 91.99% |
| 19 | `rov_cable` | 389 | 93.22% |
| 20 | `cup_plastic` | 329 | 94.27% |
| 21 | `clothing_fiber` | 298 | 95.21% |
| 22 | `wreckage_metal` | 265 | 96.05% |
| 23 | `snack_wrapper_plastic` | 172 | 96.6% |
| 24 | `boot_rubber` | 161 | 97.11% |
| 25 | `pipe_plastic` | 155 | 97.6% |
| 26 | `cable_metal` | 149 | 98.07% |
| 27 | `cup_ceramic` | 123 | 98.46% |
| 28 | `container_plastic` | 84 | 98.73% |
| 29 | `jar_glass` | 62 | 98.92% |
| 30 | `container_middle_size_metal` | 60 | 99.11% |
| 31 | `rov_bluerov` | 58 | 99.3% |
| 32 | `sanitaries_plastic` | 55 | 99.47% |
| 33 | `rov_tortuga` | 55 | 99.65% |
| 34 | `tarp_plastic` | 31 | 99.74% |
| 35 | `lid_plastic` | 20 | 99.81% |
| 36 | `animal_starfish` | 17 | 99.86% |
| 37 | `furniture_wood` | 15 | 99.91% |
| 38 | `cardboard_paper` | 13 | 99.95% |
| 39 | `rope_plastic` | 8 | 99.97% |
| 40 | `snack_wrapper_paper` | 8 | 100.0% |

### 6.1 Instances by material group (informative roll-up)

| Group | Instances | Share |
|---|---:|---:|
| animal | 13318 | 42.2% |
| plastic | 3957 | 12.5% |
| rubber | 2717 | 8.6% |
| glass | 2393 | 7.6% |
| fiber | 2337 | 7.4% |
| metal | 1604 | 5.1% |
| cement | 1404 | 4.4% |
| other | 1195 | 3.8% |
| rov_part | 963 | 3.1% |
| clay | 606 | 1.9% |
| flora | 472 | 1.5% |
| wood | 445 | 1.4% |
| ceramic | 123 | 0.4% |
| paper | 21 | 0.1% |

---

## 7. Train / Val / Test distribution

| Split | Directory | Images | Labels | Status |
|---|---|---:|---:|---|
| train | - | 0 | 0 | **DOES NOT EXIST** |
| val | - | 0 | 0 | **DOES NOT EXIST** |
| test | - | 0 | 0 | **DOES NOT EXIST** |

Searches performed, all negative:

- No `train/`, `val/`, `valid/`, `validation/` or `test/` directory anywhere in the tree.
- No `*.yaml` / `*.yml` dataset definition file.
- No `*.txt` label file (the only `.txt` is the upstream `README.txt`).
- 0 of 8,610 image paths contain a split keyword.
- No per-image label file, so there is no train/val/test label count to report.

**The dataset is 100% unsplit.** All 8,610 images and 31,555 annotations sit in one pool.

Per-subset annotation coverage (every image in every subset is annotated):

| Subset | Images | Images with annotations | Images without annotations |
|---|---:|---:|---:|
| Bistrina/Bluerobotics HD | 1390 | 1390 | 0 |
| Bistrina/Paralenz Vaquita Gen 2 | 2069 | 2069 | 0 |
| Bistrina/SIP-E323CV | 193 | 193 | 0 |
| Jakljan/Bluerobotics HD | 241 | 241 | 0 |
| Jakljan/Paralenz Vaquita | 65 | 65 | 0 |
| Lokrum/Bluerobotics HD | 556 | 556 | 0 |
| Lokrum/Paralenz Vaquita Gen 2 | 77 | 77 | 0 |
| Lokrum/SIP-E323CV | 339 | 339 | 0 |
| Marseille/SIP-E323CV | 3441 | 3441 | 0 |
| Slano/Bluerobotics HD | 168 | 168 | 0 |
| Slano/Paralenz Vaquita | 71 | 71 | 0 |
| **TOTAL** | **8610** | **8610** | **0** |

---

## 8. Data quality issues

### 8.1 Integrity checks - all clean

| # | Check | Result |
|---:|---|---|
| 14 | Images without labels | **0** - every image has >= 1 annotation (100% coverage) |
| 15 | Labels without corresponding images | **0** - single JSON, referential integrity holds |
| 16 | Empty label files | **0** - no per-image label files exist |
| 17 | Invalid class IDs | **0** - all 31,555 `category_id`s are in 1..40 |
| 18 | Invalid bounding boxes | **0** - all boxes well-formed, positive w/h, fully inside 1920x1080 |
| 19 | Corrupted / unreadable images | **0** - 8,610 / 8,610 decode cleanly (`PIL.verify()`) |
| - | Zero-byte files | **0** |
| - | Images in JSON missing on disk | **0** |
| - | Orphan images on disk (not in JSON) | **0** |
| - | Duplicate image IDs / annotation IDs | **0** / **0** |
| - | Duplicate filenames in JSON | **0** - 8,610 unique names |
| - | Boxes overflowing the frame | **0** (max overflow 0 px) |
| - | `iscrowd` regions | **0** |
| 23 | Exact duplicate images (MD5, bytewise) | **0** |

Verbatim box-integrity evidence: `nonpositive_w_or_h = 0`, `negative_xy = 0`, `x_plus_w_exceeds_image_width = 0`, `y_plus_h_exceeds_image_height = 0`, `area_field_ne_bbox_area = 0`. Box extent range: w in [1, 1920] px, h in [1, 1080] px.

### 8.2 Issues that DO exist

**I-1 - Not in YOLO format (blocking).** 31,555 boxes live inside `dataset.json` in COCO pixel space. Ultralytics needs per-image `.txt` files with normalised `cx,cy,w,h` and 0-based class indices. No `dataset.yaml` exists either.

**I-2 - No train/val/test split (blocking).** Nothing is split, so there is no basis for honest validation or model selection.

**I-3 - Severe near-duplicate frame leakage risk (critical for evaluation validity).** This dataset is ROV *video* cut into individual frames. Perceptual hashing (dHash 64-bit, Hamming <= 6) found **53,913 near-duplicate pairs** spanning **4,515 images (52.4% of the dataset)**.

Consecutive-frame similarity per subset (adjacent filenames = adjacent video frames):

| Subset | Adjacent frame pairs | Mean Hamming | Median Hamming | % of pairs <= 6 |
|---|---:|---:|---:|---:|
| Bistrina/Bluerobotics HD | 1389 | 13.91 | 14 | 10.8% |
| Bistrina/Paralenz Vaquita Gen 2 | 2068 | 9.82 | 9.0 | 32.83% |
| Bistrina/SIP-E323CV | 192 | 9.25 | 7.0 | 46.35% |
| Jakljan/Bluerobotics HD | 240 | 16.52 | 16.0 | 9.58% |
| Jakljan/Paralenz Vaquita | 64 | 16.75 | 17.0 | 6.25% |
| Lokrum/Bluerobotics HD | 555 | 15.91 | 16 | 22.88% |
| Lokrum/Paralenz Vaquita Gen 2 | 76 | 9.37 | 7.0 | 40.79% |
| Lokrum/SIP-E323CV | 338 | 10.91 | 11.0 | 28.7% |
| Marseille/SIP-E323CV | 3440 | 1.74 | 1.0 | 98.05% |
| Slano/Bluerobotics HD | 167 | 24.77 | 25 | 0.6% |
| Slano/Paralenz Vaquita | 70 | 10.47 | 10.0 | 22.86% |

`Marseille / SIP-E323CV` is effectively a single continuous dive: mean Hamming **1.74** and **98.05%** of adjacent frame pairs are near-identical, with only **1.19 objects per image** and only **5 of 40 classes** present. A naive `random_split` would put adjacent frames on both sides of the boundary and report a near-perfect mAP that reflects nothing but memorised background. Splits must be **block-wise**, never image-wise.

**I-4 - Severe class imbalance.** Max:min ratio is **807.75:1** (`animal_urchin` = 6462 vs `rope_plastic` = 8).

| Tier | Classes | Names |
|---|---:|---|
| Ultra-rare (< 10 instances) | 2 | `rope_plastic` (8), `snack_wrapper_paper` (8) |
| Rare (< 50 instances) | 7 | `tarp_plastic` (31), `furniture_wood` (15), `lid_plastic` (20), `cardboard_paper` (13), `rope_plastic` (8), `snack_wrapper_paper` (8), `animal_starfish` (17) |
| Under 100 instances | 13 | `tarp_plastic`, `container_plastic`, `container_middle_size_metal`, `sanitaries_plastic`, `jar_glass`, `rov_tortuga`, `furniture_wood`, `lid_plastic`, `cardboard_paper`, `rope_plastic`, `snack_wrapper_paper`, `rov_bluerov`, `animal_starfish` |
| Top-5 classes alone | 5 | `animal_urchin` (6462), `animal_etc` (3749), `tire_rubber` (2556), `bottle_glass` (2331), `rope_fiber` (2039) = 54.3% of all instances |

Because frames are near-duplicates, the *effective* diversity of a rare class is far lower than its raw instance count suggests. The 8 instances of `rope_plastic` may all come from a handful of adjacent frames of one dive.

**I-5 - Domain shift across the 11 subsets.** They are not IID samples of one pool:

| Subset | Inst/img | Classes present | Dominant classes |
|---|---:|---:|---|
| Bistrina / Bluerobotics HD | 5.03 | 28/40 | animal_urchin, animal_etc, rope_fiber |
| Bistrina / Paralenz Vaquita Gen 2 | 6.54 | 30/40 | animal_urchin, animal_etc, bottle_glass |
| Bistrina / SIP-E323CV | 3.89 | 18/40 | animal_urchin, animal_etc, rope_fiber |
| Jakljan / Bluerobotics HD | 3.29 | 18/40 | bottle_glass, brick_clay, can_metal |
| Jakljan / Paralenz Vaquita | 2.52 | 7/40 | bottle_glass, rov_cable, rov_bluerov |
| Lokrum / Bluerobotics HD | 4.46 | 24/40 | rov_vehicle_leg, can_metal, bottle_plastic |
| Lokrum / Paralenz Vaquita Gen 2 | 11.86 | 13/40 | animal_fish, bottle_plastic, rope_fiber |
| Lokrum / SIP-E323CV | 2.04 | 13/40 | bottle_plastic, clothing_fiber, snack_wrapper_plastic |
| Marseille / SIP-E323CV | 1.19 | 5/40 | tire_rubber, tube_cement, bag_plastic |
| Slano / Bluerobotics HD | 5.55 | 23/40 | bottle_glass, unknown_instance, plant |
| Slano / Paralenz Vaquita | 3.15 | 7/40 | bottle_glass, jar_glass, clothing_fiber |

Points of concern:

- `Marseille / SIP-E323CV` holds **40.0% of all images** but only 5/40 classes (`tire_rubber`, `tube_cement`, `bag_plastic`, `rov_cable`, `bottle_plastic`). A random split makes this single dive dominate training.
- 3 subsets are too small to split safely: `Jakljan/Paralenz Vaquita` (65 images), `Lokrum/Paralenz Vaquita Gen 2` (77 images), `Slano/Paralenz Vaquita` (71 images).
- Camera models differ (4 distinct cameras), so colour response, FOV, resolution scaling and turbidity differ. This is a genuine domain shift that a random split hides.
- 6 class(es) appear in only one subset: `animal_starfish`, `cable_metal`, `cardboard_paper`, `furniture_wood`, `rope_plastic`, `rov_vehicle_leg`.

**I-6 - Small-object dominated.** Directly constrains the real-time ROV design.

| Object size (box side) | Instances | Share |
|---|---:|---:|
| tiny_side<16px | 319 | 1.0% |
| small_16-32px | 1,133 | 3.6% |
| small_32-64px | 5,436 | 17.2% |
| medium_64-128px | 8,730 | 27.7% |
| large_128-256px | 7,289 | 23.1% |
| verylarge_>256px | 8,648 | 27.4% |

- Median box side **130.1 px**, p05 = 33.2 px, p95 = 629.2 px.
- **21.83%** of objects have a side < 64 px.
- **53.4%** of instances cover < 1% of the frame; **10.59%** cover < 0.1%.
- 465 instances have a side < 16 px (249 below 4 px). After a 640x360 real-time downscale these become sub-pixel-to-few-pixel targets, so a small-object strategy (P2 head, tiling/SAHI, or 1280+ input) is required.

---

## 9. Missing / corrupted data

**None.** This is the strongest part of the audit.

| Item | Missing / corrupted |
|---|---:|
| Corrupted or truncated images | 0 |
| Zero-byte files | 0 |
| Files < 1 KB | 0 |
| Images listed in JSON but absent on disk | 0 |
| Images on disk absent from JSON | 0 |
| Annotations pointing at a non-existent image | 0 |
| Annotations with a non-existent category | 0 |
| Images with zero annotations | 0 |
| Empty / zero-area boxes | 0 |
| Boxes exceeding image bounds | 0 |

The only genuine gaps are *structural*, not content-related: there are no YOLO `.txt` labels, no `dataset.yaml`, and no train/val/test directories.

---

## 10. Class imbalance

| Metric | Value |
|---|---:|
| Most frequent class | `animal_urchin` (6,462) |
| Least frequent class | `rope_plastic` (8) |
| Max : min ratio | **807.75:1** |
| Mean instances / class | 788.88 |
| Median instances / class | 313.5 |
| Classes with 0 instances | 0 |
| Classes with < 50 instances | 7 / 40 |
| Classes with < 10 instances | 2 / 40 |

Interpretation for this project:

1. A **single 40-class detector** is what the dataset natively supports, but mAP will be dominated by `animal_urchin`, `animal_etc`, `bottle_glass`, `tire_rubber`, `tube_cement`.
2. For an **operational ROV debris-detection** system, a **core-class model** over the well-populated, human-hazardous debris classes is far more defensible, with the full 40-class model reported as a secondary result. Recommended core set = classes with >= 100 instances.
3. `unknown_instance` (1,195) is a catch-all label and should probably be excluded from a core class set - it is not an actionable object category.
4. Mitigations for the tail: class-weighted or focal loss, copy-paste / mosaic augmentation, oversampling of rare-class frames, and evaluating per-class AP rather than only mAP50-95.

---

## 11. YOLO compatibility

### Verdict: **NOT directly compatible with Ultralytics YOLO - conversion required**

**What already satisfies YOLO requirements:**

- All 8,610 images decode cleanly and are uniformly 1920x1080 RGB JPEG.
- All 31,555 boxes are valid, positive-area, and fully inside the image.
- All `category_id` values are within the declared 40 classes; no crowd regions.
- No missing/orphan/duplicate images, so no manual repair is needed before export.
- Polygon `segmentation` is present for every annotation, enabling a future YOLO-seg run.

**Blocking items:**

| # | Blocker | Required action |
|---:|---|---|
| B1 | Annotations are COCO JSON, not per-image YOLO `.txt` | Export 8,610 label files (one line per object, 31,555 lines total) |
| B2 | `bbox` in absolute pixels `[x,y,w,h]` | Convert to **normalised** `cx,cy,w,h` (`cx=(x+w/2)/1920`, `cy=(y+h/2)/1080`, `w/=1920`, `h/=1080`) |
| B3 | Class IDs are **1..40** | Remap to **0..39** (`coco_id - 1`); YOLO requires contiguous 0-based indices or it errors/mis-indexes `names` |
| B4 | No `dataset.yaml` | Write one with `path`, `train`, `val`, `names` in the 40-name COCO ID order |
| B5 | No train/val/test split | Create a leakage-aware, block-wise, class-stratified split |
| B6 | Filenames not unique across subsets (`1.jpg` repeats) | Flatten to `<subset>__<name>.jpg` (or mirror the `<Site>/<Camera>/` structure) so every output filename is unique |
| B7 | Split must respect near-duplicate frames | Group adjacent frames into temporal blocks and split on blocks (see I-3) |

Conversion effort is low and fully deterministic: one pass over 31,555 known-good boxes, with no manual annotation work. **The data itself is not the problem; the packaging and the split strategy are.**

---

## 12. Recommended next step

> **Do not train yet.** The correct next step is a non-destructive conversion-and-packaging stage, followed by a leakage-aware split, then a baseline training run.

### Stage A - YOLO conversion (writes only to a NEW folder; source stays untouched)

1. Emit `yolo_dataset/images/{train,val,test}/` and `yolo_dataset/labels/{train,val,test}/`.
2. Per annotation write `yolo_id cx cy w h`, with `yolo_id = coco_id - 1` and normalised centroid/extent. Do not touch `dataset.json` or the original `<Site>/<Camera>/` folders.
3. Prefix output filenames with the subset id so names stay unique.
4. Write `yolo_dataset/dataset.yaml` with the 40 class names in COCO ID order.

### Stage B - Leakage-aware split (the part that must not be rushed)

1. Sort images by subset, then by frame index; cut each subset into **contiguous temporal blocks** (e.g. 25-50 frames), so no near-duplicate group straddles a split.
2. Assign whole blocks to train/val/test with a class-stratified greedy allocator targeting roughly 70 / 15 / 15, while keeping every block intact.
3. For genuine real-world generalisation evidence, additionally reserve **one or more whole site-camera subsets** as a held-out cross-domain test set (a leave-one-site-out protocol). This directly measures transfer to an unseen ROV camera and water type, which is the real deployment concern.
4. Never let two images with dHash distance <= 6 land in different splits - verify with a post-split leakage check and report the number of violating pairs (target: 0).

### Stage C - Pre-training validation

1. Re-run this audit on the converted output and assert parity: 8,610 images, 31,555 label lines, 40 classes, 0 invalid rows.
2. Run `YOLO(...)` dataset integrity check / one dry-run dataloader pass to confirm `dataset.yaml` paths resolve.

### Stage D - Baseline

1. Train **YOLOv8/YOLO11 small** baseline on the split, single class set first (`nc=40`).
2. Given the small-object profile, evaluate input size (640 vs 960/1280) and a small-object-aware variant; report FPS separately from accuracy to substantiate the **real-time ROV** claim on the deployment GPU.
3. Report per-class AP plus mAP50 and mAP50-95; report the core-class subset separately from the full 40-class set.
4. Report the cross-domain (held-out subset) results as the headline generalisation number.

### Explicitly out of scope for this stage (not done)

- No model training.
- No annotation conversion or preprocessing performed.
- No file renamed, moved, deleted or modified inside the dataset.
- No additional dataset downloaded.
- No subsets merged - all 11 `<Site>/<Camera>/` groups kept separate.

---

## Appendix - companion file

`dataset_summary.json` in the workspace root contains the same findings in machine-readable form, including the full class table with YOLO IDs, per-subset detail, per-class subset coverage, and the complete quality-check result set.
