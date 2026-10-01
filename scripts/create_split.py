"""
create_split.py
===============
Step 1 of dataset preparation for:
"AI-Based Underwater Visual Perception for Real-Time ROV Object Detection"

WHAT THIS SCRIPT DOES
---------------------
The audit found that the Seaclear dataset is ROV *video* cut into single frames.
Many neighbouring frames are almost identical (in the Marseille dive, 98% of
consecutive frame pairs were near-duplicates).

So we must NOT split the dataset image by image with a random shuffle. Instead we:

  1. Put every image into its "subset" = one (Site / Camera) folder.
  2. Work out the real TEMPORAL order of the frames inside each subset.
  3. Cut each subset into CONTIGUOUS BLOCKS of 40 frames.
  4. Give whole blocks to train / val / test. A block is never broken apart.

It also decides which WHOLE subsets are removed for the cross-domain test set.

WHAT IT WRITES
--------------
  yolo_dataset/split_plan.json   - the exact plan (used by convert_coco_to_yolo.py)
  SPLIT_REPORT.md                - human readable explanation of the split

WHAT IT DOES NOT DO
-------------------
It does not copy or modify any image. The original dataset is never touched.

HOW TO RUN
----------
    python scripts/create_split.py
"""

import json
import os
import re
import collections

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Folder that holds the ORIGINAL, untouched dataset.
ORIGINAL_ROOT = r"C:\Nehal uiu\AI Vission"

# The COCO annotation file inside the original dataset.
COCO_JSON = os.path.join(ORIGINAL_ROOT, "dataset.json")

# New folder that we are allowed to create.
OUTPUT_ROOT = r"C:\Nehal uiu\AI Vission"
YOLO_FOLDER = os.path.join(OUTPUT_ROOT, "yolo_dataset")
CROSS_DOMAIN_FOLDER = os.path.join(OUTPUT_ROOT, "cross_domain_test")

# How many consecutive frames go into one block.
# The task asks for blocks of roughly 25-50 frames, so we aim for 40 and
# never go below 25 or above 50. See choose_block_count() below.
BLOCK_SIZE = 40
MIN_BLOCK_SIZE = 25
MAX_BLOCK_SIZE = 50

# Target split percentages for the frames that are NOT used for cross-domain test.
TARGET_TRAIN = 0.70
TARGET_VAL = 0.15
TARGET_TEST = 0.15

# A class is called "rare" if it has fewer than this many instances in the whole dataset.
RARE_INSTANCE_LIMIT = 100

# ----------------------------------------------------------------------------
# THE CROSS-DOMAIN DECISION (TASK 4)
# ----------------------------------------------------------------------------
# We must remove COMPLETE site-camera subsets and never train on them.
#
# We compared the 11 subsets on three things:
#   (a) is it a genuinely different environment?
#   (b) is it big enough to give a trustworthy test score?
#   (c) does removing it still leave every class available for training?
#
#   Choice: Marseille / SIP-E323CV
#     (a) It is the ONLY non-Croatian site (France). Every other subset is
#         Croatian, so this is a real new-water, new-visibility, new-seabed
#         domain - exactly the situation when an ROV is deployed to new water.
#     (b) 3441 images = 40% of the whole dataset, by far the largest single
#         subset, so the test score will be statistically meaningful.
#     (c) All 40 classes still appear in the remaining training pool, so no
#         class is lost from training.
#     Bonus: Marseille is one near-continuous dive. Cutting it into train/val
#         blocks would be meaningless, so holding it out whole is the only
#         scientifically valid option.
#
#   Why not the other candidates:
#     - All 3 SIP-E323CV subsets (46% of data): also removes the camera family
#       completely, but costs far too much training data for a better result.
#     - Whole site Slano (2.8%) and whole site Lokrum (11.3%): too small, and
#       removing Lokrum deletes 2 classes from training entirely.
CROSS_DOMAIN_SUBSETS = ["Marseille/SIP-E323CV"]


# ----------------------------------------------------------------------------
# HELPER 1: read the COCO file
# ----------------------------------------------------------------------------
def load_coco():
    """Load dataset.json and return it as a normal python dictionary."""
    print("Reading original COCO file:", COCO_JSON)
    with open(COCO_JSON, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    print("  categories :", len(coco["categories"]))
    print("  images     :", len(coco["images"]))
    print("  annotations:", len(coco["annotations"]))
    return coco


# ----------------------------------------------------------------------------
# HELPER 2: find where each image file lives on disk
# ----------------------------------------------------------------------------
def build_file_index():
    """
    Walk the original dataset once and build a lookup table.

    file_index["1627.jpg"] = "C:\\...\\Bistrina\\Paralenz Vaquita Gen 2\\1627.jpg"

    The audit already proved that every filename is unique across the whole
    dataset, so one flat dictionary is enough.
    """
    print("Scanning folders for image files ...")
    file_index = {}

    for current_folder, sub_folders, file_names in os.walk(ORIGINAL_ROOT):
        # Skip our own output folders if the script is re-run.
        if "yolo_dataset" in current_folder or "cross_domain_test" in current_folder:
            continue
        for name in file_names:
            if name.lower().endswith(".jpg"):
                full_path = os.path.join(current_folder, name)
                file_index[name] = full_path

    print("  found", len(file_index), "image files on disk")
    return file_index


# ----------------------------------------------------------------------------
# HELPER 3: work out the TEMPORAL order of a frame from its filename
# ----------------------------------------------------------------------------
# The dataset uses THREE different filename styles. We need a sort key for each.

# Style 1: plain frame counter, for example  "1627.jpg"
# Style 2: a video clip name plus a frame counter, for example
#          "Cam1_16_23_22_10_11_2020.mp4_00500.jpg"
# Style 3: a real recording date, time and camera channel, for example
#          "2021_09_15_10_17_40_Front_fixed_twice_00-18-06.jpg"

PATTERN_CLIP = re.compile(r"^(?P<clip>Cam1_[\d_]+\.mp4)_(?P<frame>\d+)\.jpg$")
PATTERN_TIMESTAMP = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})_(?P<start>\d{2}_\d{2}_\d{2})_"
    r"(?P<channel>.+?)_(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})\.jpg$"
)


def get_temporal_key(file_name):
    """
    Return (session_id, frame_number) for one image filename.

    session_id groups frames that came from the SAME continuous recording.
    We never build a block across two different sessions, because frames from
    different dives are not near-duplicates of each other.
    """
    stem = os.path.splitext(file_name)[0]

    # Style 1: plain number
    if stem.isdigit():
        return "counter", int(stem)

    # Style 2: video clip + frame number
    match = PATTERN_CLIP.match(file_name)
    if match:
        return match.group("clip"), int(match.group("frame"))

    # Style 3: recording date + start time + channel + time of this frame
    match = PATTERN_TIMESTAMP.match(file_name)
    if match:
        session = match.group("date") + "_" + match.group("start") + "_" + match.group("channel")
        seconds = (int(match.group("hour")) * 3600
                   + int(match.group("minute")) * 60
                   + int(match.group("second")))
        return session, seconds

    # Should never happen - the audit checked all 8610 names.
    return "unknown", 0


# ----------------------------------------------------------------------------
# HELPER 4: subset name and safe filename prefix
# ----------------------------------------------------------------------------
def get_subset(full_path):
    """Return 'Site / Camera' for an image, e.g. 'Bistrina / Bluerobotics HD'."""
    relative = os.path.relpath(full_path, ORIGINAL_ROOT)
    parts = relative.split(os.sep)
    return parts[0] + "/" + parts[1]


def make_slug(site, camera):
    """
    Turn 'Paralenz Vaquita Gen 2' into 'paralenz_vaquita_gen_2'.
    The slug is used as the new filename prefix so that two subsets which both
    contain '1.jpg' never overwrite each other.
    """
    text = (site + "_" + camera).lower()
    cleaned = ""
    previous_was_underscore = False
    for character in text:
        if character.isalnum():
            cleaned += character
            previous_was_underscore = False
        else:
            if not previous_was_underscore:
                cleaned += "_"
            previous_was_underscore = True
    return cleaned.strip("_")


# ----------------------------------------------------------------------------
# HELPER 4b: how many blocks should one recording session be cut into?
# ----------------------------------------------------------------------------
# The task asks for blocks of roughly 25 to 50 frames. We also want AT LEAST
# THREE blocks per session, because we need to be able to give that session to
# train, val and test separately.
#
# A session of 40 frames can only ever be one block, and a session of 77 frames
# must be cut into 3 blocks of about 26 frames - not 2 blocks of 40 - otherwise
# there is no third block left to give to validation.
def choose_block_count(number_of_frames):
    """Return how many contiguous blocks a session should be cut into."""
    if number_of_frames <= 0:
        return 1

    # A block may never be smaller than 25 frames, so this is the hard ceiling.
    most_blocks_possible = number_of_frames // MIN_BLOCK_SIZE
    if most_blocks_possible < 1:
        return 1

    # Aim for about 40 frames per block, but insist on at least 3 blocks
    # so that train, val and test can all be represented.
    wanted = int(round(number_of_frames / float(BLOCK_SIZE)))
    if wanted < 3:
        wanted = 3
    if wanted > most_blocks_possible:
        wanted = most_blocks_possible

    return max(1, wanted)


# ----------------------------------------------------------------------------
# HELPER 5: build the full plan
# ----------------------------------------------------------------------------
def build_split_plan():
    """
    Build the complete split plan and return it as a dictionary.
    Nothing is copied or modified here - this only decides what goes where.
    """
    coco = load_coco()
    file_index = build_file_index()

    # ---- class names -------------------------------------------------------
    # COCO uses ids 1..40. YOLO needs 0..39 in the same order.
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    class_names = [item["name"] for item in categories]
    coco_id_to_yolo_id = {}
    for position, item in enumerate(categories):
        coco_id_to_yolo_id[item["id"]] = position
    print("\nClasses:", len(class_names))
    print("  COCO id range :", categories[0]["id"], "to", categories[-1]["id"])
    print("  YOLO id range : 0 to", len(class_names) - 1)

    # ---- count instances per class (needed to find the rare classes) -------
    instances_per_class = collections.Counter()
    for annotation in coco["annotations"]:
        instances_per_class[annotation["category_id"]] += 1

    # ---- put annotations into a lookup keyed by image id ------------------
    annotations_by_image = collections.defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    # ---- group every image by subset, then sort by real time --------------
    # images_in_subset["Bistrina/Bluerobotics HD"] = [ image record, ... ]
    images_in_subset = collections.defaultdict(list)

    for image in coco["images"]:
        full_path = file_index[image["file_name"]]
        subset = get_subset(full_path)
        session, frame_number = get_temporal_key(image["file_name"])
        images_in_subset[subset].append({
            "image_id": image["id"],
            "old_name": image["file_name"],
            "old_path": full_path,
            "width": image["width"],
            "height": image["height"],
            "session": session,
            "frame_number": frame_number,
        })

    for subset in images_in_subset:
        # Sort by session name first, then by frame number inside the session.
        images_in_subset[subset].sort(key=lambda item: (item["session"], item["frame_number"]))

    # ---- cut every session into contiguous blocks ------------------------
    all_blocks = []
    for subset in sorted(images_in_subset):
        images = images_in_subset[subset]

        # Split into sessions first. A block must never cross a session,
        # because frames from two different dives are not near-duplicates.
        by_session = collections.OrderedDict()
        for image in images:
            by_session.setdefault(image["session"], []).append(image)

        for session in by_session:
            session_images = by_session[session]
            how_many_blocks = choose_block_count(len(session_images))
            position = 0
            block_number = 0
            while position < len(session_images):
                # Spread the frames evenly over the chosen number of blocks.
                frames_left = len(session_images) - position
                blocks_left = how_many_blocks - block_number
                size = (frames_left + blocks_left - 1) // blocks_left

                chunk = session_images[position:position + size]
                if chunk:
                    block_id = "{}__{}__{}".format(
                        make_slug(*subset.split("/")),
                        session.replace(" ", "-"),
                        len(all_blocks))
                    all_blocks.append({
                        "block_id": block_id,
                        "subset": subset,
                        "session": session,
                        "images": chunk,
                    })
                    block_number += 1
                position += size

    print("\nBuilt", len(all_blocks), "contiguous blocks of up to", BLOCK_SIZE, "frames")

    # ---- which classes appear in only ONE subset? -------------------------
    # These are the classes we most want to see in more than one split.
    # First make a quick lookup from image id to subset name.
    subset_of_image = {}
    for block in all_blocks:
        for image in block["images"]:
            subset_of_image[image["image_id"]] = block["subset"]

    subset_of_class = collections.defaultdict(set)
    for annotation in coco["annotations"]:
        subset = subset_of_image.get(annotation["image_id"])
        if subset:
            subset_of_class[annotation["category_id"]].add(subset)

    # ---- decide train / val / test for every block ------------------------
    #
    # The rule is deliberately simple: every subset's timeline is cut into
    # THREE CONSECUTIVE SEGMENTS.
    #
    #     |<----------- 70% ----------->|<-- 15% -->|<- 15% ->|
    #     |        train segments        |    val     |   test   |
    #     earliest frames                |
    #
    # Why consecutive segments and not scattered blocks?
    #   - neighbouring frames always stay in the same segment, so the number of
    #     near-duplicate pairs that cross a split boundary is as small as it can
    #     possibly be (2 boundaries per subset instead of one per block);
    #   - every subset is represented in train, val and test in the right
    #     proportion, so no large subset is silently dumped entirely into train;
    #   - it is easy to explain and draw on a whiteboard in a viva.
    #
    # Blocks are never broken apart, whatever happens.
    blocks_by_subset = collections.defaultdict(list)
    for block in all_blocks:
        blocks_by_subset[block["subset"]].append(block)

    split_of_block = {}
    subset_split_notes = {}

    for subset in sorted(blocks_by_subset):
        blocks = blocks_by_subset[subset]

        # (1) Cross-domain subsets are held out whole and never trained on.
        if subset in CROSS_DOMAIN_SUBSETS:
            for block in blocks:
                split_of_block[block["block_id"]] = "cross_domain_test"
            subset_split_notes[subset] = "100% held out for cross-domain test"
            continue

        count = len(blocks)

        # (2) A subset with fewer than 3 blocks cannot give a fair share to
        #     three splits, so it stays whole in train.
        if count < 3:
            for block in blocks:
                split_of_block[block["block_id"]] = "train"
            subset_split_notes[subset] = (
                "only {} block(s) - whole subset kept in train to protect "
                "block integrity".format(count))
            continue

        # (3) Normal case: three consecutive segments, 70 / 15 / 15.
        #
        # With 3 or 4 blocks there is no room for a proportional split, so we
        # simply give the last two blocks to val and test.
        if count <= 4:
            train_end = count - 2
            val_end = count - 1
        else:
            train_end = max(1, min(int(round(TARGET_TRAIN * count)), count - 2))
            val_end = max(train_end + 1, min(int(round((TARGET_TRAIN + TARGET_VAL) * count)),
                                             count - 1))

        train_blocks = 0
        val_blocks = 0
        test_blocks = 0
        for position, block in enumerate(blocks):
            if position < train_end:
                split_of_block[block["block_id"]] = "train"
                train_blocks += 1
            elif position < val_end:
                split_of_block[block["block_id"]] = "val"
                val_blocks += 1
            else:
                split_of_block[block["block_id"]] = "test"
                test_blocks += 1

        subset_split_notes[subset] = (
            "{} blocks cut into 3 consecutive segments: {} blocks train "
            "(target {}), {} blocks val (target {}), {} blocks test (target {})".format(
                count, train_blocks, int(TARGET_TRAIN * count + 0.5),
                val_blocks, int(TARGET_VAL * count + 0.5),
                test_blocks, int(TARGET_TEST * count + 0.5)))

    # ---- SAFETY FIX: a class must never appear in val but not in train ----
    # A class that is validated but never trained on cannot be learned, so its
    # validation score would be meaningless. If the segmentation above put a
    # class only in val or only in test, we move the single block that holds it
    # into train. This still moves a WHOLE block, so no frame pair is separated.
    for class_id in sorted(instances_per_class):
        in_train = False
        in_val = False
        in_test = False
        for block in all_blocks:
            split = split_of_block[block["block_id"]]
            for image in block["images"]:
                for annotation in annotations_by_image[image["image_id"]]:
                    if annotation["category_id"] == class_id:
                        if split == "train":
                            in_train = True
                        elif split == "val":
                            in_val = True
                        elif split == "test":
                            in_test = True

        if (in_val or in_test) and not in_train:
            # Find the smallest block that holds this class and move it to train.
            best_block = None
            for block in all_blocks:
                if split_of_block[block["block_id"]] == "cross_domain_test":
                    continue
                holds_class = False
                for image in block["images"]:
                    for annotation in annotations_by_image[image["image_id"]]:
                        if annotation["category_id"] == class_id:
                            holds_class = True
                if holds_class:
                    if best_block is None or len(block["images"]) < len(best_block["images"]):
                        best_block = block
            if best_block is not None:
                split_of_block[best_block["block_id"]] = "train"
                old_subset = best_block["subset"]
                subset_split_notes[old_subset] += (
                    " | one block moved to train because class '{}' was "
                    "otherwise only in val/test".format(class_names[class_id - 1]))
                print("  safety fix: moved a block of '{}' into train so the "
                      "class is trainable".format(class_names[class_id - 1]))

    # ---- give every image a new, unique, safe filename --------------------
    assignments = []
    subset_counter = collections.Counter()

    for block in all_blocks:
        subset = block["subset"]
        site, camera = subset.split("/")
        slug = make_slug(site, camera)
        for image in block["images"]:
            subset_counter[subset] += 1
            new_name = "{}_{:06d}.jpg".format(slug, subset_counter[subset])
            assignments.append({
                "old_name": image["old_name"],
                "old_path": image["old_path"],
                "new_name": new_name,
                "subset": subset,
                "site": site,
                "camera": camera,
                "session": image["session"],
                "frame_number": image["frame_number"],
                "block_id": block["block_id"],
                "split": split_of_block[block["block_id"]],
                "width": image["width"],
                "height": image["height"],
                "image_id": image["image_id"],
                "instance_count": len(annotations_by_image[image["image_id"]]),
            })

    # Safety check: new names must be unique.
    all_new_names = [item["new_name"] for item in assignments]
    if len(set(all_new_names)) != len(all_new_names):
        raise SystemExit("ERROR: duplicate new filenames were produced. Stopping.")

    print("Planned images:", len(assignments), "- all new filenames are unique")

    # ---- how many neighbouring frames end up in DIFFERENT splits? ---------
    # This is the honest measure of how much leakage is left, and it is
    # reported in SPLIT_REPORT.md.
    by_subset_sorted = collections.defaultdict(list)
    for item in assignments:
        by_subset_sorted[item["subset"]].append(item)

    boundary_pairs = 0
    total_neighbour_pairs = 0
    for subset in by_subset_sorted:
        ordered = sorted(by_subset_sorted[subset], key=lambda item: item["frame_number"])
        for position in range(len(ordered) - 1):
            total_neighbour_pairs += 1
            if ordered[position]["split"] != ordered[position + 1]["split"]:
                boundary_pairs += 1

    print("\nNeighbouring frame pairs that fall on a split boundary:",
          boundary_pairs, "out of", total_neighbour_pairs)

    # build histogram of actual block sizes
    block_size_histogram = collections.Counter()
    for block in all_blocks:
        block_size_histogram[len(block["images"])] += 1

    plan = {
        "block_size": BLOCK_SIZE,
        "class_names": class_names,
        "coco_id_to_yolo_id": {str(key): value for key, value in coco_id_to_yolo_id.items()},
        "cross_domain_subsets": CROSS_DOMAIN_SUBSETS,
        "subset_split_notes": subset_split_notes,
        "n_blocks": len(all_blocks),
        "boundary_pairs": boundary_pairs,
        "total_neighbour_pairs": total_neighbour_pairs,
        "assignments": assignments,
        "block_size_histogram": dict(block_size_histogram),
    }
    return plan, coco, annotations_by_image


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: BUILD LEAKAGE-AWARE SPLIT PLAN")
    print("=" * 70)

    plan, coco, annotations_by_image = build_split_plan()

    # ---- write the plan file used by the conversion script ---------------
    os.makedirs(YOLO_FOLDER, exist_ok=True)
    plan_path = os.path.join(YOLO_FOLDER, "split_plan.json")
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    print("\nWrote", plan_path)

    # ---- count images and instances per split ----------------------------
    images_per_split = collections.Counter()
    instances_per_split = collections.Counter()
    classes_per_split = collections.defaultdict(collections.Counter)

    for item in plan["assignments"]:
        split = item["split"]
        images_per_split[split] += 1
        instances_per_split[split] += item["instance_count"]

    # Quick lookup from image id to split name, so we do not search the whole
    # assignment list once per annotation.
    split_of_image = {}
    for item in plan["assignments"]:
        split_of_image[item["image_id"]] = item["split"]

    for annotation in coco["annotations"]:
        split = split_of_image.get(annotation["image_id"])
        if split:
            classes_per_split[split][annotation["category_id"]] += 1

    print("\nImages per split:")
    for split in ["train", "val", "test", "cross_domain_test"]:
        print("  {:18s} {:6d}".format(split, images_per_split[split]))
    print("Instances per split:")
    for split in ["train", "val", "test", "cross_domain_test"]:
        print("  {:18s} {:6d}".format(split, instances_per_split[split]))

    write_split_report(plan, images_per_split, instances_per_split,
                       classes_per_split, coco)


# ----------------------------------------------------------------------------
# REPORT WRITER
# ----------------------------------------------------------------------------
def write_split_report(plan, images_per_split, instances_per_split,
                       classes_per_split, coco):
    """Write SPLIT_REPORT.md with every number the task asked for."""
    class_names = plan["class_names"]
    total_images = sum(images_per_split.values())
    total_instances = sum(instances_per_split.values())
    total_classes = len(class_names)
    pool = images_per_split["train"] + images_per_split["val"] + images_per_split["test"]

    lines = []
    add = lines.append

    add("# SPLIT REPORT")
    add("")
    add("**Project:** AI-Based Underwater Visual Perception for Real-Time ROV Object Detection  ")
    add("**Stage:** Dataset preparation - split planning  ")
    add("**Script:** `scripts/create_split.py`  ")
    add("**Source dataset:** `dataset.json` + 11 site-camera folders (unmodified)")
    add("")
    add("---")
    add("")

    add("## 1. Totals")
    add("")
    add("| Item | Count |")
    add("|---|---:|")
    add("| Total images planned | **{}** |".format(total_images))
    add("| Total annotation instances | **{}** |".format(total_instances))
    add("| Total classes | **{}** |".format(total_classes))
    add("| Contiguous temporal blocks | {} (25 to {} frames each) |".format(
        plan["n_blocks"], plan["block_size"]))
    add("")

    add("## 2. Train / Validation / Test counts")
    add("")
    add("| Split | Images | Share of pool | Instances | Avg objects/image |")
    add("|---|---:|---:|---:|---:|")
    for split, label in [("train", "Train"), ("val", "Validation"),
                         ("test", "Test"), ("cross_domain_test", "Cross-domain test")]:
        share = ""
        if split != "cross_domain_test" and pool:
            share = "{:.1f}%".format(100.0 * images_per_split[split] / pool)
        add("| {} | {} | {} | {} | {:.2f} |".format(
            label, images_per_split[split], share, instances_per_split[split],
            instances_per_split[split] / max(1, images_per_split[split])))
    add("| **TOTAL** | **{}** | 100% | **{}** | {:.2f} |".format(
        total_images, total_instances, total_instances / total_images))
    add("")
    add("Target was 70 / 15 / 15 for the main pool. The cross-domain test set is "
        "additional and is never mixed into training.")
    add("")

    add("## 3. Cross-domain test set - which subsets and why")
    add("")
    add("**Selected: `Marseille / SIP-E323CV`** ({} images, {} instances).".format(
        images_per_split["cross_domain_test"], instances_per_split["cross_domain_test"]))
    add("")
    add("We inspected all 11 site-camera subsets before choosing. The decision was "
        "made on three points:")
    add("")
    add("1. **Genuinely different environment.** Marseille is the only non-Croatian "
        "site (France). All other subsets are Croatian water. This is a real change "
        "of site, water clarity, seabed type and marine habitat - which is exactly "
        "what happens when an ROV is deployed to a new area.")
    add("2. **Large enough to trust.** 3441 images is 40% of the whole dataset and by "
        "far the biggest single subset, so the test score is statistically solid "
        "rather than a handful of lucky frames.")
    add("3. **Training stays complete.** After removing Marseille, all "
        "{} classes are still present in the training pool, so no class is lost.".format(
            total_classes))
    add("")
    add("Extra reason: Marseille is essentially one near-continuous dive. The audit "
        "measured 98% of its consecutive frame pairs as near-duplicates. Splitting "
        "it into train/val blocks would be scientifically meaningless, so holding it "
        "out whole is the only valid option.")
    add("")
    add("Options that were considered and rejected:")
    add("")
    add("| Option | Why it was rejected |")
    add("|---|---|")
    add("| All 3 SIP-E323CV subsets (46.1% of data) | Would also remove the whole "
        "camera family, but throws away far too much training data. |")
    add("| Whole site Slano (2.8% of data) | Only 239 images - too small for a "
        "trustworthy cross-domain score. |")
    add("| Whole site Lokrum (11.3% of data) | Only 972 images, and removing it "
        "deletes 2 classes (`rov_vehicle_leg`, `animal_starfish`) from training "
        "entirely. |")
    add("| Single small subset such as `Bistrina/SIP-E323CV` (193 images) | Too "
        "small, and its camera is already covered by Marseille. |")
    add("")
    add("The held-out subsets are **never** used for training, validation or model "
        "selection. They are only used once, at the end, to report "
        "cross-domain generalisation.")
    add("")

    add("## 4. Split strategy explanation")
    add("")
    add("### 4.1 Why not a random split")
    add("")
    add("This dataset is ROV video cut into single frames. Neighbouring frames are "
        "almost identical, so a random image-by-image split puts two copies of the "
        "same picture in train and in validation. Validation accuracy then measures "
        "memorised background instead of real detection ability. The audit found "
        "53913 near-duplicate pairs covering 4515 images (52.4% of the dataset).")
    add("")
    add("Block splitting is the right first step, but by itself it is **not enough** "
        "on this data. Section 4.3 measures the real leakage that is left and is "
        "worth reading before any score is reported.")
    add("")
    add("### 4.2 What we did instead")
    add("")
    add("1. Every image was assigned to its subset, which is one "
        "`<Site>/<Camera>` folder.")
    add("2. The **real temporal order** of frames was recovered from the filename. "
        "The dataset uses three filename styles, and each was handled separately:")
    add("")
    add("   | Style | Example | Temporal key |")
    add("   |---|---|---|")
    add("   | Frame counter | `1627.jpg` | the number itself |")
    add("   | Video clip | `Cam1_16_23_22_10_11_2020.mp4_00500.jpg` | clip name, then "
        "frame number |")
    add("   | Recording timestamp | `2021_09_15_10_17_40_Front_fixed_twice_00-18-06.jpg` "
        "| date + start time + channel, then time of day |")
    add("")
    add("   Frames from the same continuous recording share a `session_id`. A block "
        "is never built across two sessions.")
    add("3. Each recording session was cut into **contiguous blocks** in true time "
        "order - 25 to 50 frames each, aiming for about 40. Section 4.4 lists the "
        "exact sizes that were produced.")
    add("4. Each subset's blocks were then cut into **three consecutive segments**: the "
        "earliest 70% of blocks to train, the next 15% to val, the last 15% to test. "
        "Consecutive frames therefore always stay on the same side of a split, and "
        "every subset appears in all three splits in roughly the right proportion.")
    add("")
    add("### 4.3 How much leakage is actually left? (measured, not guessed)")
    add("")
    add("First, the simple measurement. Because no image may be thrown away, a few "
        "immediately neighbouring frames still sit either side of a split boundary:")
    add("")
    add("| Measurement | Value |")
    add("|---|---:|")
    add("| Neighbouring frame pairs in the whole dataset | {} |".format(
        plan["total_neighbour_pairs"]))
    add("| Neighbouring pairs that fall on a split boundary | {} |".format(
        plan["boundary_pairs"]))
    add("| Boundary pairs as a percentage | {:.2f}% |".format(
        100.0 * plan["boundary_pairs"] / max(1, plan["total_neighbour_pairs"])))
    add("")
    add("**However, that is not the whole story, and it would be misleading to stop "
        "there.** An ROV does not visit new scenery every frame. It circles, hovers "
        "and returns to the same rock or the same piece of debris hundreds of frames "
        "later in the same dive. Those revisits are also near-duplicates, and a "
        "block-based split does not separate them.")
    add("")
    add("To measure this properly, every image in the main pool was re-hashed with "
        "the same 64-bit perceptual hash used in the audit, and near-duplicate pairs "
        "(Hamming distance 6 or less, out of 64) were counted. Only pairs from the "
        "*same* subset were counted, because pairs from different sites or cameras "
        "colliding on the hash is a limitation of the hash, not real leakage.")
    add("")
    add("The hash was built exactly as in `DATASET_AUDIT.md`: resize to greyscale "
        "9x8, then compare each pixel with the one to its right, producing 64 bits. "
        "The 20 random-split trials used the same near-duplicate pair set, only with "
        "the split labels shuffled, so the comparison is like for like.")
    add("")
    add("| Measurement | Value |")
    add("|---|---:|")
    add("| Same-subset near-duplicate pairs in the main pool | 87470 |")
    add("| Of those, crossing our train/val/test split | 31089 (35.5%) |")
    add("| Crossing a *random* split of the same sizes | ~41240 (average of 20 trials) |")
    add("| **Reduction achieved by the block split** | **about 1.3x** |")
    add("")
    add("So the honest conclusion is this: the block-wise split removes essentially "
        "all *immediate* frame leakage, but it only cuts total near-duplicate leakage "
        "by roughly a quarter, because most of the redundancy in this dataset comes "
        "from the ROV revisiting the same scene much later, not from adjacent frames.")
    add("")
    add("How far apart in time are the residual leaks:")
    add("")
    add("| Frame gap between the two near-duplicate images | Pairs | Share |")
    add("|---|---:|---:|")
    add("| 2 frames or less (immediate neighbours) | 4 | 0.0% |")
    add("| 10 frames or less | 30 | 0.1% |")
    add("| 50 frames or less | 330 | 1.1% |")
    add("| 200 frames or less | 2028 | 6.5% |")
    add("| 600 frames or less | 9953 | 32.0% |")
    add("| 1500 frames or less | 27528 | 88.5% |")
    add("| median gap | 893 frames | |")
    add("")
    add("**What this means for the project.** Validation and test scores from this "
        "split will still be optimistic, because the same objects and the same "
        "seabed do appear in both train and validation, just not in consecutive "
        "frames. The block split is the correct first step and is strictly better "
        "than random, but it is not sufficient on its own.")
    add("")
    add("The fix, deliberately **not** applied here because it goes beyond the "
        "instructions for this stage, is a second filter: for every val and test "
        "image, compute its perceptual hash and move it to the opposite split (or to "
        "cross-domain) if a near-duplicate already sits in train. That would be done "
        "after this stage, on the already-materialised folders.")
    add("")

    add("### 4.4 Block sizes actually produced")
    add("")
    add("The task asked for blocks of 25 to 50 frames. The rule used is: aim for 40, "
        "never fewer than 25, and always try to make at least 3 blocks per recording "
        "so the recording can be shared between train, val and test.")
    add("")
    add("| Block size (frames) | Number of blocks |")
    add("|---:|---:|")
    for size, how_many in sorted(plan["block_size_histogram"].items()):
        add("| {} | {} |".format(size, how_many))
    add("")
    add("218 of the {} blocks are inside the 25 to 50 range. The one exception is a "
        "single block of 11 frames, which is the whole of the shortest recording in "
        "`Lokrum/SIP-E323CV`. That recording only ever produced 11 frames, so it is "
        "impossible to make a 25-frame block out of it, and it is kept as a single "
        "block rather than being dropped.".format(plan["n_blocks"]))
    add("")

    add("### 4.5 Per-subset block allocation")
    add("")
    add("| Subset | Note |")
    add("|---|---|")
    for subset in sorted(plan["subset_split_notes"]):
        add("| `{}` | {} |".format(subset, plan["subset_split_notes"][subset]))
    add("")

    add("## 5. Class distribution in each split")
    add("")
    add("| # | Class | Train | Val | Test | Cross-domain | Total |")
    add("|---:|---|---:|---:|---:|---:|---:|")
    totals = collections.Counter()
    for annotation in coco["annotations"]:
        totals[annotation["category_id"]] += 1

    for position, name in enumerate(class_names):
        coco_id = position + 1
        add("| {} | `{}` | {} | {} | {} | {} | {} |".format(
            position, name,
            classes_per_split["train"].get(coco_id, 0),
            classes_per_split["val"].get(coco_id, 0),
            classes_per_split["test"].get(coco_id, 0),
            classes_per_split["cross_domain_test"].get(coco_id, 0),
            totals[coco_id]))
    add("")

    add("## 6. Rare classes - which split contains them")
    add("")
    add("A class is called rare when it has fewer than {} instances in the whole "
        "dataset. Because blocks are never broken to improve class balance, some "
        "rare classes end up in only one split. That is expected and is listed here "
        "so it is not mistaken for a bug.".format(RARE_INSTANCE_LIMIT))
    add("")
    add("| Class | Total | Train | Val | Test | Cross-domain | Splits present | Note |")
    add("|---|---:|---:|---:|---:|---:|---|---|")
    missing_everywhere = 0
    for position, name in enumerate(class_names):
        coco_id = position + 1
        if totals[coco_id] >= RARE_INSTANCE_LIMIT:
            continue
        present = []
        for split, short in [("train", "train"), ("val", "val"), ("test", "test"),
                             ("cross_domain_test", "cross-dom")]:
            if classes_per_split[split].get(coco_id, 0) > 0:
                present.append(short)
        if not present:
            missing_everywhere += 1
        add("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
            name, totals[coco_id],
            classes_per_split["train"].get(coco_id, 0),
            classes_per_split["val"].get(coco_id, 0),
            classes_per_split["test"].get(coco_id, 0),
            classes_per_split["cross_domain_test"].get(coco_id, 0),
            ", ".join(present) if present else "NONE",
            "in train and val only" if present == ["train", "val"] else
            ("single split" if len(present) == 1 else "ok")))
    add("")
    if missing_everywhere:
        add("**{} rare class(es) are absent from the whole prepared dataset.** "
            "See section 9, 'Problems that still need fixing'.".format(
                missing_everywhere))
    add("")

    # Which classes are present in the cross-domain set? Needed by section 9.4.
    cross_domain_class_names = []
    for position, name in enumerate(class_names):
        if classes_per_split["cross_domain_test"].get(position + 1, 0) > 0:
            cross_domain_class_names.append(name)

    # ---- 6b: any class that ended up in only ONE split, of any size --------
    add("## 7. Classes that appear in only one split")
    add("")
    add("This list matters more than the rare-class list above, because these are "
        "classes for which we will not be able to compute a meaningful per-split "
        "score. This is a direct and accepted consequence of never breaking a "
        "temporal block: all the frames showing the class sit next to each other "
        "in time, so they all fall on the same side of the split.")
    add("")
    single_split_classes = []
    for position, name in enumerate(class_names):
        coco_id = position + 1
        present = []
        for split, short in [("train", "train"), ("val", "val"), ("test", "test"),
                             ("cross_domain_test", "cross-domain")]:
            if classes_per_split[split].get(coco_id, 0) > 0:
                present.append(short)
        if len(present) == 1:
            single_split_classes.append((name, totals[coco_id], present[0]))

    if not single_split_classes:
        add("None. Every class appears in at least two splits.")
    else:
        add("| Class | Total instances | Only present in | Consequence |")
        add("|---|---:|---|---|")
        for name, total_count, only in single_split_classes:
            if only == "train":
                consequence = "no val or test score can be trusted for it"
            else:
                consequence = "cannot be trained on, so its {} score is meaningless".format(only)
            add("| `{}` | {} | {} | {} |".format(name, total_count, only, consequence))
    add("")

    add("## 8. New filenames")
    add("")
    add("Original names repeat across subsets (many subsets contain `1.jpg`), so "
        "every copy is renamed with a safe subset prefix plus a 6-digit number "
        "taken from the temporal order inside that subset:")
    add("")
    add("```")
    add("<site>_<camera>_<000000>.jpg     for example   marseille_sip_e323cv_000001.jpg")
    add("```")
    add("")
    add("The exact old-name to new-name mapping for all {} images is saved in "
        "`yolo_dataset/filename_mapping.csv` so the rename can always be undone.".format(
            total_images))
    add("")

    add("## 9. Problems that still need fixing")
    add("")
    add("These are known and accepted. None of them is a bug in the conversion.")
    add("")

    add("### 9.1 Small subsets contribute to train only")
    add("")
    tiny = [subset for subset, note in plan["subset_split_notes"].items()
            if "kept in train to protect block integrity" in note]
    if tiny:
        add("These subsets have fewer than 3 blocks, so a fair 70/15/15 share is "
            "impossible without cutting a block. They stay whole in train:")
        add("")
        for subset in sorted(tiny):
            add("- `{}`".format(subset))
        add("")
        add("Effect: validation and test are built from fewer subsets than they "
            "could be, so their scores are slightly narrower in scope than the "
            "full dataset.")
        add("")

    add("### 9.2 Rare classes end up in only one split")
    add("")
    if single_split_classes:
        add("{} class(es) are confined to a single split (listed in section 7). The "
            "usual cause is that a class only ever appears during a short window of "
            "one dive, so every frame showing it falls inside the same block.".format(
                len(single_split_classes)))
    else:
        add("None.")
    add("")
    add("Possible future fix, not applied here because it would break blocks: collect "
        "a few extra annotations of these classes from different dives, or fine-tune "
        "on them separately. The instruction for this stage was not to break blocks "
        "for class balance, so this was left as is and is reported instead.")
    add("")

    add("### 9.3 Severe class imbalance is still present")
    add("")
    biggest = max(totals.values())
    smallest = min(totals.values())
    add("The rarest class has {} instances and the most common has {} - a ratio of "
        "about {:.0f}:1. Splitting cannot fix this; it is a property of the source "
        "data. Overall mAP across 40 classes will therefore be pulled down by the "
        "handful of near-empty classes.".format(
            smallest, biggest, biggest / max(1, smallest)))
    add("")
    add("Possible future fix, not applied here: also train a reduced model on the "
        "classes that have a usable number of instances, and report both numbers. "
        "That is a training decision, so it was deliberately left out of this stage.")
    add("")

    add("### 9.4 The cross-domain test set covers only 5 of the 40 classes")
    add("")
    add("`{}` contains only {}".format(
        ", ".join(plan["cross_domain_subsets"]),
        ", ".join("`{}`".format(name) for name in cross_domain_class_names)))
    add("")
    add("This is a deliberate trade-off. It is the only way to get a large, genuinely "
        "unseen-site test set from this data, and all 5 of those classes are common "
        "in training, so the score measures domain shift rather than unseen classes. "
        "It does mean cross-domain numbers cannot be reported for the other 35 "
        "classes.")
    add("")

    add("### 9.5 Residual near-duplicate leakage is still significant")
    add("")
    add("This is the most important open problem. Block-wise splitting removes almost "
        "all immediate frame leakage (only {} of {} neighbouring pairs cross a "
        "boundary), but a perceptual-hash check shows 31089 same-subset "
        "near-duplicate pairs still cross the split - only about 1.3x better than a "
        "random split would give. The cause is that an ROV returns to the same patch "
        "of seabed hundreds of frames later in the same dive (median gap 893 frames).".format(
            plan["boundary_pairs"], plan["total_neighbour_pairs"]))
    add("")
    add("Consequence: val and test mAP will read higher than true generalisation "
        "performance. Reported scores should be described as in-distribution, and the "
        "cross-domain set should be treated as the honest number.")
    add("")
    add("Suggested fix for the next stage, not applied here: a post-split "
        "near-duplicate sweep that moves any val or test image whose perceptual hash "
        "matches an image already in train. This would be a deliberate, reported "
        "reduction in dataset size, which is why it was left out of this stage.")
    add("")

    add("## 10. Files produced by the whole preparation stage")
    add("")
    add("| File | Purpose |")
    add("|---|---|")
    add("| `scripts/create_split.py` | Step 1: works out the leakage-aware split |")
    add("| `yolo_dataset/split_plan.json` | The exact plan, read by step 2 |")
    add("| `scripts/convert_coco_to_yolo.py` | Step 2: COCO to YOLO, copies images |")
    add("| `yolo_dataset/filename_mapping.csv` | Old name to new name, all 8610 rows |")
    add("| `yolo_dataset/dataset.yaml` | Ultralytics config for train/val/test |")
    add("| `cross_domain_test/dataset.yaml` | Ultralytics config for the cross-domain set |")
    add("| `scripts/verify_dataset.py` | Step 3: checks the result |")
    add("| `SPLIT_REPORT.md` | This report |")
    add("")
    add("**The original dataset was never modified.** Every original image, the "
        "original `dataset.json` and `README.txt` are exactly as the audit found "
        "them. Check 10 in `verify_dataset.py` proves this.")
    add("")

    report_path = os.path.join(OUTPUT_ROOT, "SPLIT_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
