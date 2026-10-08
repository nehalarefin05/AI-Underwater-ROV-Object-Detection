# SPLIT REPORT

**Project:** AI-Based Underwater Visual Perception for Real-Time ROV Object Detection  
**Stage:** Dataset preparation - split planning  
**Script:** `scripts/create_split.py`  
**Source dataset:** `dataset.json` + 11 site-camera folders (unmodified)

---

## 1. Totals

| Item | Count |
|---|---:|
| Total images planned | **8610** |
| Total annotation instances | **31555** |
| Total classes | **40** |
| Contiguous temporal blocks | 219 (25 to 40 frames each) |

## 2. Train / Validation / Test counts

| Split | Images | Share of pool | Instances | Avg objects/image |
|---|---:|---:|---:|---:|
| Train | 3574 | 69.1% | 17920 | 5.01 |
| Validation | 854 | 16.5% | 4971 | 5.82 |
| Test | 741 | 14.3% | 4570 | 6.17 |
| Cross-domain test | 3441 |  | 4094 | 1.19 |
| **TOTAL** | **8610** | 100% | **31555** | 3.66 |

Target was 70 / 15 / 15 for the main pool. The cross-domain test set is additional and is never mixed into training.

## 3. Cross-domain test set - which subsets and why

**Selected: `Marseille / SIP-E323CV`** (3441 images, 4094 instances).

We inspected all 11 site-camera subsets before choosing. The decision was made on three points:

1. **Genuinely different environment.** Marseille is the only non-Croatian site (France). All other subsets are Croatian water. This is a real change of site, water clarity, seabed type and marine habitat - which is exactly what happens when an ROV is deployed to a new area.
2. **Large enough to trust.** 3441 images is 40% of the whole dataset and by far the biggest single subset, so the test score is statistically solid rather than a handful of lucky frames.
3. **Training stays complete.** After removing Marseille, all 40 classes are still present in the training pool, so no class is lost.

Extra reason: Marseille is essentially one near-continuous dive. The audit measured 98% of its consecutive frame pairs as near-duplicates. Splitting it into train/val blocks would be scientifically meaningless, so holding it out whole is the only valid option.

Options that were considered and rejected:

| Option | Why it was rejected |
|---|---|
| All 3 SIP-E323CV subsets (46.1% of data) | Would also remove the whole camera family, but throws away far too much training data. |
| Whole site Slano (2.8% of data) | Only 239 images - too small for a trustworthy cross-domain score. |
| Whole site Lokrum (11.3% of data) | Only 972 images, and removing it deletes 2 classes (`rov_vehicle_leg`, `animal_starfish`) from training entirely. |
| Single small subset such as `Bistrina/SIP-E323CV` (193 images) | Too small, and its camera is already covered by Marseille. |

The held-out subsets are **never** used for training, validation or model selection. They are only used once, at the end, to report cross-domain generalisation.

## 4. Split strategy explanation

### 4.1 Why not a random split

This dataset is ROV video cut into single frames. Neighbouring frames are almost identical, so a random image-by-image split puts two copies of the same picture in train and in validation. Validation accuracy then measures memorised background instead of real detection ability. The audit found 53913 near-duplicate pairs covering 4515 images (52.4% of the dataset).

Block splitting is the right first step, but by itself it is **not enough** on this data. Section 4.3 measures the real leakage that is left and is worth reading before any score is reported.

### 4.2 What we did instead

1. Every image was assigned to its subset, which is one `<Site>/<Camera>` folder.
2. The **real temporal order** of frames was recovered from the filename. The dataset uses three filename styles, and each was handled separately:

   | Style | Example | Temporal key |
   |---|---|---|
   | Frame counter | `1627.jpg` | the number itself |
   | Video clip | `Cam1_16_23_22_10_11_2020.mp4_00500.jpg` | clip name, then frame number |
   | Recording timestamp | `2021_09_15_10_17_40_Front_fixed_twice_00-18-06.jpg` | date + start time + channel, then time of day |

   Frames from the same continuous recording share a `session_id`. A block is never built across two sessions.
3. Each recording session was cut into **contiguous blocks** in true time order - 25 to 50 frames each, aiming for about 40. Section 4.4 lists the exact sizes that were produced.
4. Each subset's blocks were then cut into **three consecutive segments**: the earliest 70% of blocks to train, the next 15% to val, the last 15% to test. Consecutive frames therefore always stay on the same side of a split, and every subset appears in all three splits in roughly the right proportion.

### 4.3 How much leakage is actually left? (measured, not guessed)

First, the simple measurement. Because no image may be thrown away, a few immediately neighbouring frames still sit either side of a split boundary:

| Measurement | Value |
|---|---:|
| Neighbouring frame pairs in the whole dataset | 8599 |
| Neighbouring pairs that fall on a split boundary | 54 |
| Boundary pairs as a percentage | 0.63% |

**However, that is not the whole story, and it would be misleading to stop there.** An ROV does not visit new scenery every frame. It circles, hovers and returns to the same rock or the same piece of debris hundreds of frames later in the same dive. Those revisits are also near-duplicates, and a block-based split does not separate them.

To measure this properly, every image in the main pool was re-hashed with the same 64-bit perceptual hash used in the audit, and near-duplicate pairs (Hamming distance 6 or less, out of 64) were counted. Only pairs from the *same* subset were counted, because pairs from different sites or cameras colliding on the hash is a limitation of the hash, not real leakage.

The hash was built exactly as in `DATASET_AUDIT.md`: resize to greyscale 9x8, then compare each pixel with the one to its right, producing 64 bits. The 20 random-split trials used the same near-duplicate pair set, only with the split labels shuffled, so the comparison is like for like.

| Measurement | Value |
|---|---:|
| Same-subset near-duplicate pairs in the main pool | 87470 |
| Of those, crossing our train/val/test split | 31089 (35.5%) |
| Crossing a *random* split of the same sizes | ~41240 (average of 20 trials) |
| **Reduction achieved by the block split** | **about 1.3x** |

So the honest conclusion is this: the block-wise split removes essentially all *immediate* frame leakage, but it only cuts total near-duplicate leakage by roughly a quarter, because most of the redundancy in this dataset comes from the ROV revisiting the same scene much later, not from adjacent frames.

How far apart in time are the residual leaks:

| Frame gap between the two near-duplicate images | Pairs | Share |
|---|---:|---:|
| 2 frames or less (immediate neighbours) | 4 | 0.0% |
| 10 frames or less | 30 | 0.1% |
| 50 frames or less | 330 | 1.1% |
| 200 frames or less | 2028 | 6.5% |
| 600 frames or less | 9953 | 32.0% |
| 1500 frames or less | 27528 | 88.5% |
| median gap | 893 frames | |

**What this means for the project.** Validation and test scores from this split will still be optimistic, because the same objects and the same seabed do appear in both train and validation, just not in consecutive frames. The block split is the correct first step and is strictly better than random, but it is not sufficient on its own.

The fix, deliberately **not** applied here because it goes beyond the instructions for this stage, is a second filter: for every val and test image, compute its perceptual hash and move it to the opposite split (or to cross-domain) if a near-duplicate already sits in train. That would be done after this stage, on the already-materialised folders.

### 4.4 Block sizes actually produced

The task asked for blocks of 25 to 50 frames. The rule used is: aim for 40, never fewer than 25, and always try to make at least 3 blocks per recording so the recording can be shared between train, val and test.

| Block size (frames) | Number of blocks |
|---:|---:|
| 11 | 1 |
| 25 | 1 |
| 26 | 2 |
| 29 | 1 |
| 30 | 2 |
| 32 | 1 |
| 33 | 2 |
| 34 | 1 |
| 35 | 1 |
| 36 | 1 |
| 37 | 1 |
| 38 | 5 |
| 39 | 33 |
| 40 | 135 |
| 41 | 28 |
| 42 | 4 |

218 of the 219 blocks are inside the 25 to 50 range. The one exception is a single block of 11 frames, which is the whole of the shortest recording in `Lokrum/SIP-E323CV`. That recording only ever produced 11 frames, so it is impossible to make a 25-frame block out of it, and it is kept as a single block rather than being dropped.

### 4.5 Per-subset block allocation

| Subset | Note |
|---|---|
| `Bistrina/Bluerobotics HD` | 35 blocks cut into 3 consecutive segments: 24 blocks train (target 25), 6 blocks val (target 5), 5 blocks test (target 5) |
| `Bistrina/Paralenz Vaquita Gen 2` | 52 blocks cut into 3 consecutive segments: 36 blocks train (target 36), 8 blocks val (target 8), 8 blocks test (target 8) |
| `Bistrina/SIP-E323CV` | 6 blocks cut into 3 consecutive segments: 4 blocks train (target 4), 1 blocks val (target 1), 1 blocks test (target 1) |
| `Jakljan/Bluerobotics HD` | 6 blocks cut into 3 consecutive segments: 4 blocks train (target 4), 1 blocks val (target 1), 1 blocks test (target 1) |
| `Jakljan/Paralenz Vaquita` | only 2 block(s) - whole subset kept in train to protect block integrity |
| `Lokrum/Bluerobotics HD` | 14 blocks cut into 3 consecutive segments: 10 blocks train (target 10), 2 blocks val (target 2), 2 blocks test (target 2) |
| `Lokrum/Paralenz Vaquita Gen 2` | 3 blocks cut into 3 consecutive segments: 1 blocks train (target 2), 1 blocks val (target 0), 1 blocks test (target 0) | one block moved to train because class 'animal_starfish' was otherwise only in val/test |
| `Lokrum/SIP-E323CV` | 9 blocks cut into 3 consecutive segments: 6 blocks train (target 6), 2 blocks val (target 1), 1 blocks test (target 1) |
| `Marseille/SIP-E323CV` | 100% held out for cross-domain test |
| `Slano/Bluerobotics HD` | 4 blocks cut into 3 consecutive segments: 2 blocks train (target 3), 1 blocks val (target 1), 1 blocks test (target 1) |
| `Slano/Paralenz Vaquita` | only 2 block(s) - whole subset kept in train to protect block integrity |

## 5. Class distribution in each split

| # | Class | Train | Val | Test | Cross-domain | Total |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `can_metal` | 737 | 206 | 187 | 0 | 1130 |
| 1 | `tarp_plastic` | 22 | 0 | 9 | 0 | 31 |
| 2 | `container_plastic` | 16 | 44 | 24 | 0 | 84 |
| 3 | `bottle_plastic` | 763 | 213 | 184 | 101 | 1261 |
| 4 | `tube_cement` | 331 | 8 | 34 | 1031 | 1404 |
| 5 | `plant` | 395 | 45 | 32 | 0 | 472 |
| 6 | `container_middle_size_metal` | 54 | 4 | 2 | 0 | 60 |
| 7 | `animal_etc` | 2755 | 404 | 590 | 0 | 3749 |
| 8 | `animal_sponge` | 793 | 267 | 50 | 0 | 1110 |
| 9 | `bottle_glass` | 1799 | 219 | 313 | 0 | 2331 |
| 10 | `wreckage_metal` | 261 | 4 | 0 | 0 | 265 |
| 11 | `unknown_instance` | 899 | 154 | 142 | 0 | 1195 |
| 12 | `pipe_plastic` | 131 | 21 | 3 | 0 | 155 |
| 13 | `net_plastic` | 926 | 33 | 1 | 0 | 960 |
| 14 | `animal_shells` | 925 | 50 | 20 | 0 | 995 |
| 15 | `rope_fiber` | 1347 | 375 | 317 | 0 | 2039 |
| 16 | `animal_urchin` | 2607 | 1946 | 1909 | 0 | 6462 |
| 17 | `cup_plastic` | 96 | 114 | 119 | 0 | 329 |
| 18 | `brick_clay` | 481 | 52 | 73 | 0 | 606 |
| 19 | `bag_plastic` | 94 | 90 | 30 | 668 | 882 |
| 20 | `sanitaries_plastic` | 37 | 12 | 6 | 0 | 55 |
| 21 | `clothing_fiber` | 201 | 86 | 11 | 0 | 298 |
| 22 | `cup_ceramic` | 51 | 40 | 32 | 0 | 123 |
| 23 | `boot_rubber` | 153 | 8 | 0 | 0 | 161 |
| 24 | `tire_rubber` | 131 | 108 | 194 | 2123 | 2556 |
| 25 | `jar_glass` | 33 | 10 | 19 | 0 | 62 |
| 26 | `rov_cable` | 84 | 18 | 116 | 171 | 389 |
| 27 | `rov_tortuga` | 14 | 31 | 10 | 0 | 55 |
| 28 | `branch_wood` | 276 | 139 | 15 | 0 | 430 |
| 29 | `furniture_wood` | 15 | 0 | 0 | 0 | 15 |
| 30 | `snack_wrapper_plastic` | 60 | 81 | 31 | 0 | 172 |
| 31 | `lid_plastic` | 3 | 2 | 15 | 0 | 20 |
| 32 | `cardboard_paper` | 13 | 0 | 0 | 0 | 13 |
| 33 | `rope_plastic` | 8 | 0 | 0 | 0 | 8 |
| 34 | `cable_metal` | 149 | 0 | 0 | 0 | 149 |
| 35 | `animal_fish` | 883 | 102 | 0 | 0 | 985 |
| 36 | `snack_wrapper_paper` | 4 | 0 | 4 | 0 | 8 |
| 37 | `rov_vehicle_leg` | 317 | 77 | 67 | 0 | 461 |
| 38 | `rov_bluerov` | 47 | 0 | 11 | 0 | 58 |
| 39 | `animal_starfish` | 9 | 8 | 0 | 0 | 17 |

## 6. Rare classes - which split contains them

A class is called rare when it has fewer than 100 instances in the whole dataset. Because blocks are never broken to improve class balance, some rare classes end up in only one split. That is expected and is listed here so it is not mistaken for a bug.

| Class | Total | Train | Val | Test | Cross-domain | Splits present | Note |
|---|---:|---:|---:|---:|---:|---|---|
| `tarp_plastic` | 31 | 22 | 0 | 9 | 0 | train, test | ok |
| `container_plastic` | 84 | 16 | 44 | 24 | 0 | train, val, test | ok |
| `container_middle_size_metal` | 60 | 54 | 4 | 2 | 0 | train, val, test | ok |
| `sanitaries_plastic` | 55 | 37 | 12 | 6 | 0 | train, val, test | ok |
| `jar_glass` | 62 | 33 | 10 | 19 | 0 | train, val, test | ok |
| `rov_tortuga` | 55 | 14 | 31 | 10 | 0 | train, val, test | ok |
| `furniture_wood` | 15 | 15 | 0 | 0 | 0 | train | single split |
| `lid_plastic` | 20 | 3 | 2 | 15 | 0 | train, val, test | ok |
| `cardboard_paper` | 13 | 13 | 0 | 0 | 0 | train | single split |
| `rope_plastic` | 8 | 8 | 0 | 0 | 0 | train | single split |
| `snack_wrapper_paper` | 8 | 4 | 0 | 4 | 0 | train, test | ok |
| `rov_bluerov` | 58 | 47 | 0 | 11 | 0 | train, test | ok |
| `animal_starfish` | 17 | 9 | 8 | 0 | 0 | train, val | in train and val only |


## 7. Classes that appear in only one split

This list matters more than the rare-class list above, because these are classes for which we will not be able to compute a meaningful per-split score. This is a direct and accepted consequence of never breaking a temporal block: all the frames showing the class sit next to each other in time, so they all fall on the same side of the split.

| Class | Total instances | Only present in | Consequence |
|---|---:|---|---|
| `furniture_wood` | 15 | train | no val or test score can be trusted for it |
| `cardboard_paper` | 13 | train | no val or test score can be trusted for it |
| `rope_plastic` | 8 | train | no val or test score can be trusted for it |
| `cable_metal` | 149 | train | no val or test score can be trusted for it |

## 8. New filenames

Original names repeat across subsets (many subsets contain `1.jpg`), so every copy is renamed with a safe subset prefix plus a 6-digit number taken from the temporal order inside that subset:

```
<site>_<camera>_<000000>.jpg     for example   marseille_sip_e323cv_000001.jpg
```

The exact old-name to new-name mapping for all 8610 images is saved in `yolo_dataset/filename_mapping.csv` so the rename can always be undone.

## 9. Problems that still need fixing

These are known and accepted. None of them is a bug in the conversion.

### 9.1 Small subsets contribute to train only

These subsets have fewer than 3 blocks, so a fair 70/15/15 share is impossible without cutting a block. They stay whole in train:

- `Jakljan/Paralenz Vaquita`
- `Slano/Paralenz Vaquita`

Effect: validation and test are built from fewer subsets than they could be, so their scores are slightly narrower in scope than the full dataset.

### 9.2 Rare classes end up in only one split

4 class(es) are confined to a single split (listed in section 7). The usual cause is that a class only ever appears during a short window of one dive, so every frame showing it falls inside the same block.

Possible future fix, not applied here because it would break blocks: collect a few extra annotations of these classes from different dives, or fine-tune on them separately. The instruction for this stage was not to break blocks for class balance, so this was left as is and is reported instead.

### 9.3 Severe class imbalance is still present

The rarest class has 8 instances and the most common has 6462 - a ratio of about 808:1. Splitting cannot fix this; it is a property of the source data. Overall mAP across 40 classes will therefore be pulled down by the handful of near-empty classes.

Possible future fix, not applied here: also train a reduced model on the classes that have a usable number of instances, and report both numbers. That is a training decision, so it was deliberately left out of this stage.

### 9.4 The cross-domain test set covers only 5 of the 40 classes

`Marseille/SIP-E323CV` contains only `bottle_plastic`, `tube_cement`, `bag_plastic`, `tire_rubber`, `rov_cable`

This is a deliberate trade-off. It is the only way to get a large, genuinely unseen-site test set from this data, and all 5 of those classes are common in training, so the score measures domain shift rather than unseen classes. It does mean cross-domain numbers cannot be reported for the other 35 classes.

### 9.5 Residual near-duplicate leakage is still significant

This is the most important open problem. Block-wise splitting removes almost all immediate frame leakage (only 54 of 8599 neighbouring pairs cross a boundary), but a perceptual-hash check shows 31089 same-subset near-duplicate pairs still cross the split - only about 1.3x better than a random split would give. The cause is that an ROV returns to the same patch of seabed hundreds of frames later in the same dive (median gap 893 frames).

Consequence: val and test mAP will read higher than true generalisation performance. Reported scores should be described as in-distribution, and the cross-domain set should be treated as the honest number.

Suggested fix for the next stage, not applied here: a post-split near-duplicate sweep that moves any val or test image whose perceptual hash matches an image already in train. This would be a deliberate, reported reduction in dataset size, which is why it was left out of this stage.

## 10. Files produced by the whole preparation stage

| File | Purpose |
|---|---|
| `scripts/create_split.py` | Step 1: works out the leakage-aware split |
| `yolo_dataset/split_plan.json` | The exact plan, read by step 2 |
| `scripts/convert_coco_to_yolo.py` | Step 2: COCO to YOLO, copies images |
| `yolo_dataset/filename_mapping.csv` | Old name to new name, all 8610 rows |
| `yolo_dataset/dataset.yaml` | Ultralytics config for train/val/test |
| `cross_domain_test/dataset.yaml` | Ultralytics config for the cross-domain set |
| `scripts/verify_dataset.py` | Step 3: checks the result |
| `SPLIT_REPORT.md` | This report |

**The original dataset was never modified.** Every original image, the original `dataset.json` and `README.txt` are exactly as the audit found them. Check 10 in `verify_dataset.py` proves this.
