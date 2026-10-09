# ============================================================
# CAPSICUM DATASET CLEANING + SPLITTING + OFFLINE AUGMENTATION
# ============================================================
# Pipeline:
# 1. Count original dataset
# 2. SHA-256 exact duplicate report
# 3. pHash near-duplicate grouping
# 4. Group-wise train/valid/test split
# 5. TRAIN ONLY offline augmentation
# 6. VALID / TEST remain original
# 7. Exactly 1100 TRAIN images/class
#
# UPDATED FOR CURRENT 13-CLASS DATASET
# ============================================================

import os
import shutil
import random
import hashlib
import csv
from collections import defaultdict

from PIL import Image, ImageEnhance, ImageOps
import imagehash

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "data new")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_dataset")
REPORT_DIR = os.path.join(BASE_DIR, "dataset_reports")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

PHASH_THRESHOLD = 8
TRAIN_TARGET = 1100

CLASSES = [
    "Anthracnose",
    "Larva",
    "Magnesium",
    "Phytophthora blight",
    "bacterial spot",
    "blossom-end rot",
    "down leaf aphid",
    "fruit thrips",
    "healthy",
    "powdery mildew",
    "snail",
    "upperleaf thrips",
    "virus",
]

EXPECTED_CLASSES = len(CLASSES)
EXPECTED_TRAIN_TOTAL = EXPECTED_CLASSES * TRAIN_TARGET

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(SEED)

# ============================================================
# FILE HELPERS
# ============================================================

def get_image_files(folder):
    if not os.path.exists(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(IMAGE_EXTENSIONS)
    )


def clean_generated_directories():
    print("\n" + "=" * 80)
    print("CLEANING GENERATED DIRECTORIES")
    print("=" * 80)

    for directory in (SPLIT_DIR, OUTPUT_DIR):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def validate_source_dataset():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Source dataset not found:\n{SOURCE_DIR}")

    missing = [c for c in CLASSES
               if not os.path.isdir(os.path.join(SOURCE_DIR, c))]

    if missing:
        raise RuntimeError("Missing classes:\n" + "\n".join(missing))

    print(f"All {EXPECTED_CLASSES} classes found.")


# ============================================================
# COUNT
# ============================================================

def count_source_dataset():
    print("\n" + "=" * 80)
    print("ORIGINAL SOURCE DATASET COUNT")
    print("=" * 80)

    counts = {}
    total = 0

    for c in CLASSES:
        n = len(get_image_files(os.path.join(SOURCE_DIR, c)))
        counts[c] = n
        total += n
        print(f"{c:<25} : {n}")

    print("-" * 80)
    print(f"{'TOTAL':<25} : {total}")
    print("=" * 80)
    return counts


# ============================================================
# ADAPTIVE AUGMENTATION LEVEL
# ============================================================
# Based on ORIGINAL class count:
# < 300       STRONG
# 300 - 500   MODERATE-STRONG
# 501 - 800   MODERATE
# 801 - 1000  LIGHT
# 1001 - 1100 VERY-LIGHT
# > 1100      NONE
#
# Current data therefore gets automatically assigned the
# appropriate level; no manual class-by-class editing needed.
# ============================================================

def get_augmentation_level(n):
    if n < 300:
        return "strong"
    if n <= 500:
        return "moderate_strong"
    if n <= 800:
        return "moderate"
    if n <= 1000:
        return "light"
    if n <= 1100:
        return "very_light"
    return "none"


def print_augmentation_levels(counts):
    print("\n" + "=" * 80)
    print("AUTOMATIC AUGMENTATION LEVEL")
    print("=" * 80)
    print(f"{'Class':<25}{'Original':<12}{'Level':<20}{'Target':<10}")
    print("-" * 80)

    for c in CLASSES:
        print(
            f"{c:<25}{counts[c]:<12}"
            f"{get_augmentation_level(counts[c]):<20}"
            f"{TRAIN_TARGET:<10}"
        )
    print("=" * 80)


def save_augmentation_report(counts):
    path = os.path.join(REPORT_DIR, "augmentation_threshold_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "original_count",
                    "augmentation_level", "train_target"])
        for c in CLASSES:
            w.writerow([
                c, counts[c], get_augmentation_level(counts[c]),
                TRAIN_TARGET
            ])
    print(f"Augmentation report saved: {path}")


# ============================================================
# EXACT DUPLICATES - SHA256
# ============================================================

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates():
    print("\n" + "=" * 80)
    print("FINDING EXACT DUPLICATES - SHA-256")
    print("=" * 80)

    hash_map = defaultdict(list)

    for c in CLASSES:
        for path in get_image_files(os.path.join(SOURCE_DIR, c)):
            try:
                hash_map[calculate_sha256(path)].append(path)
            except Exception as e:
                print(f"Hash error: {path}\n{e}")

    groups = [(h, files) for h, files in hash_map.items()
              if len(files) > 1]

    report = os.path.join(REPORT_DIR, "exact_duplicates.csv")
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "file"])
        for h, files in groups:
            for path in files:
                w.writerow([h, path])

    print(f"Exact duplicate groups: {len(groups)}")
    print(f"Report saved: {report}")
    return groups


# ============================================================
# pHASH GROUPING
# ============================================================

def calculate_phash(filepath):
    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            return imagehash.phash(img)
    except Exception:
        return None


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_phash_groups():
    print("\n" + "=" * 80)
    print("FINDING NEAR DUPLICATES - pHASH")
    print("=" * 80)

    all_groups = []
    report = os.path.join(REPORT_DIR, "phash_groups.csv")

    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "group_id", "file"])

        for c in CLASSES:
            files = get_image_files(os.path.join(SOURCE_DIR, c))
            if not files:
                continue

            hashes = [calculate_phash(x) for x in files]
            uf = UnionFind(len(files))

            for i in range(len(files)):
                if hashes[i] is None:
                    continue
                for j in range(i + 1, len(files)):
                    if hashes[j] is not None and \
                       (hashes[i] - hashes[j]) <= PHASH_THRESHOLD:
                        uf.union(i, j)

            groups = defaultdict(list)
            for i, path in enumerate(files):
                groups[uf.find(i)].append(path)

            group_list = list(groups.values())
            print(f"{c:<25} : {len(group_list)} groups")

            for gi, group in enumerate(group_list):
                group_id = f"{c}_{gi}"
                all_groups.append((c, group))
                for path in group:
                    w.writerow([c, group_id, path])

    print(f"\npHASH report saved: {report}")
    return all_groups


# ============================================================
# GROUP-WISE SPLIT
# ============================================================

def split_dataset(phash_groups):
    print("\n" + "=" * 80)
    print("GROUP-WISE DATASET SPLIT")
    print("=" * 80)

    class_groups = defaultdict(list)
    for c, group in phash_groups:
        class_groups[c].append(group)

    split_counts = {}

    for c in CLASSES:
        groups = class_groups[c].copy()
        random.shuffle(groups)

        total = sum(len(g) for g in groups)
        train_target = round(total * TRAIN_RATIO)
        valid_target = round(total * VALID_RATIO)

        train_groups, valid_groups, test_groups = [], [], []
        train_n = valid_n = 0

        for group in groups:
            size = len(group)

            if train_n + size <= train_target:
                train_groups.append(group)
                train_n += size
            elif valid_n + size <= valid_target:
                valid_groups.append(group)
                valid_n += size
            else:
                test_groups.append(group)

        # Safety: never leave training empty.
        if not train_groups and groups:
            train_groups.append(groups[0])

        for split in ("train", "valid", "test"):
            os.makedirs(os.path.join(SPLIT_DIR, split, c),
                        exist_ok=True)

        def copy_groups(groups_to_copy, split):
            destination = os.path.join(SPLIT_DIR, split, c)
            copied = 0
            for group in groups_to_copy:
                for src in group:
                    dst = os.path.join(destination, os.path.basename(src))
                    shutil.copy2(src, dst)
                    copied += 1
            return copied

        actual_train = copy_groups(train_groups, "train")
        actual_valid = copy_groups(valid_groups, "valid")
        actual_test = copy_groups(test_groups, "test")

        split_counts[c] = {
            "train": actual_train,
            "valid": actual_valid,
            "test": actual_test,
            "total": actual_train + actual_valid + actual_test,
        }

        print(
            f"{c:<25} "
            f"Train: {actual_train:<5} "
            f"Valid: {actual_valid:<5} "
            f"Test: {actual_test:<5}"
        )

    print("=" * 80)
    return split_counts


# ============================================================
# IMAGE AUGMENTATION HELPERS
# ============================================================

def random_flip(img, p=0.5):
    if random.random() < p:
        img = ImageOps.mirror(img)
    return img


def random_rotate(img, max_angle, p):
    if random.random() < p:
        angle = random.uniform(-max_angle, max_angle)
        img = img.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=None,
        )
    return img


def random_crop_resize(img, min_scale, p):
    if random.random() >= p:
        return img

    w, h = img.size
    scale = random.uniform(min_scale, 1.0)
    crop_w = max(1, int(w * scale))
    crop_h = max(1, int(h * scale))

    left = random.randint(0, max(0, w - crop_w))
    top = random.randint(0, max(0, h - crop_h))

    img = img.crop((left, top, left + crop_w, top + crop_h))
    return img.resize((w, h), Image.Resampling.LANCZOS)


def adjust(img, enhancer, low, high, p):
    if random.random() < p:
        factor = random.uniform(low, high)
        img = enhancer(img).enhance(factor)
    return img


# ============================================================
# AUGMENTATION PROFILES
# ============================================================
# Improvements:
# - moderate geometry to preserve disease/deficiency symptoms
# - saturation changes instead of aggressive color shifts
# - no extreme transformations
# - probabilities scale with data scarcity
# ============================================================

def augment_strong(img):
    img = random_crop_resize(img, 0.70, 0.85)
    img = random_flip(img, 0.50)
    img = random_rotate(img, 25, 0.75)
    img = adjust(img, ImageEnhance.Brightness, 0.75, 1.25, 0.70)
    img = adjust(img, ImageEnhance.Contrast, 0.75, 1.25, 0.65)
    img = adjust(img, ImageEnhance.Color, 0.80, 1.20, 0.55)
    img = adjust(img, ImageEnhance.Sharpness, 0.80, 1.25, 0.30)
    return img


def augment_moderate_strong(img):
    img = random_crop_resize(img, 0.75, 0.70)
    img = random_flip(img, 0.50)
    img = random_rotate(img, 22, 0.65)
    img = adjust(img, ImageEnhance.Brightness, 0.80, 1.20, 0.60)
    img = adjust(img, ImageEnhance.Contrast, 0.80, 1.20, 0.55)
    img = adjust(img, ImageEnhance.Color, 0.85, 1.15, 0.45)
    img = adjust(img, ImageEnhance.Sharpness, 0.85, 1.20, 0.25)
    return img


def augment_moderate(img):
    img = random_crop_resize(img, 0.80, 0.50)
    img = random_flip(img, 0.50)
    img = random_rotate(img, 18, 0.50)
    img = adjust(img, ImageEnhance.Brightness, 0.85, 1.15, 0.45)
    img = adjust(img, ImageEnhance.Contrast, 0.85, 1.15, 0.40)
    img = adjust(img, ImageEnhance.Color, 0.90, 1.10, 0.30)
    return img


def augment_light(img):
    img = random_crop_resize(img, 0.88, 0.30)
    img = random_flip(img, 0.50)
    img = random_rotate(img, 12, 0.35)
    img = adjust(img, ImageEnhance.Brightness, 0.92, 1.08, 0.25)
    img = adjust(img, ImageEnhance.Contrast, 0.92, 1.08, 0.20)
    img = adjust(img, ImageEnhance.Color, 0.94, 1.06, 0.15)
    return img


def augment_very_light(img):
    img = random_flip(img, 0.50)
    img = random_rotate(img, 8, 0.20)
    img = adjust(img, ImageEnhance.Brightness, 0.95, 1.05, 0.15)
    img = adjust(img, ImageEnhance.Contrast, 0.95, 1.05, 0.12)
    return img


def apply_augmentation(img, level):
    if level == "strong":
        return augment_strong(img)
    if level == "moderate_strong":
        return augment_moderate_strong(img)
    if level == "moderate":
        return augment_moderate(img)
    if level == "light":
        return augment_light(img)
    if level == "very_light":
        return augment_very_light(img)
    if level == "none":
        return img.copy()
    raise ValueError(f"Unknown augmentation level: {level}")


# ============================================================
# TRAIN AUGMENTATION
# ============================================================

def augment_training_data(source_counts):
    print("\n" + "=" * 80)
    print("TRAINING DATA AUGMENTATION")
    print("=" * 80)

    final_counts = {}

    for c in CLASSES:
        source_count = source_counts[c]
        level = get_augmentation_level(source_count)

        src_dir = os.path.join(SPLIT_DIR, "train", c)
        out_dir = os.path.join(OUTPUT_DIR, "train", c)
        os.makedirs(out_dir, exist_ok=True)

        train_files = get_image_files(src_dir)
        current = len(train_files)

        if current == 0:
            raise RuntimeError(f"No training images found for: {c}")

        print("\n" + "-" * 80)
        print(f"Class                 : {c}")
        print(f"Original count        : {source_count}")
        print(f"Train split count     : {current}")
        print(f"Augmentation level    : {level.upper()}")

        # If there are already more than target, randomly downsample
        # TRAIN ONLY so every class remains exactly 1100.
        if current > TRAIN_TARGET:
            print(f"Downsampling train split to {TRAIN_TARGET}.")
            selected = random.sample(train_files, TRAIN_TARGET)

            for src in selected:
                shutil.copy2(src, os.path.join(out_dir,
                                                os.path.basename(src)))

            final_counts[c] = TRAIN_TARGET
            continue

        # Copy original train images first.
        for src in train_files:
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

        required = TRAIN_TARGET - current
        generated = 0

        print(f"Images to generate    : {required}")

        while generated < required:
            src = random.choice(train_files)

            try:
                with Image.open(src) as original:
                    img = ImageOps.exif_transpose(original).convert("RGB")
                    aug = apply_augmentation(img, level)

                    filename = (
                        f"aug_{generated:05d}_"
                        f"{os.path.splitext(os.path.basename(src))[0]}.jpg"
                    )
                    aug.save(
                        os.path.join(out_dir, filename),
                        format="JPEG",
                        quality=95,
                        optimize=True,
                    )

                generated += 1

            except Exception as e:
                print(f"Augmentation error: {src}\n{e}")

        final = len(get_image_files(out_dir))

        if final != TRAIN_TARGET:
            raise RuntimeError(
                f"Final training count mismatch for {c}: "
                f"{final}, expected {TRAIN_TARGET}"
            )

        final_counts[c] = final
        print(f"Final train count     : {final}")

    total = sum(final_counts.values())

    print("\n" + "=" * 80)
    print("FINAL TRAINING DATA COUNT")
    print("=" * 80)

    for c in CLASSES:
        print(f"{c:<25} : {final_counts[c]}")

    print("-" * 80)
    print(f"{'TOTAL':<25} : {total}")

    if total != EXPECTED_TRAIN_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_TOTAL} train images, got {total}"
        )

    print("\nSUCCESS: Exactly 1100 images/class.")
    return final_counts


# ============================================================
# COPY VALIDATION + TEST
# ============================================================

def copy_validation_test():
    print("\n" + "=" * 80)
    print("COPYING VALIDATION + TEST DATA")
    print("=" * 80)

    for split in ("valid", "test"):
        for c in CLASSES:
            src_dir = os.path.join(SPLIT_DIR, split, c)
            out_dir = os.path.join(OUTPUT_DIR, split, c)
            os.makedirs(out_dir, exist_ok=True)

            files = get_image_files(src_dir)
            for src in files:
                shutil.copy2(src, os.path.join(out_dir,
                                               os.path.basename(src)))

            print(f"{split:<8} {c:<25} {len(files)}")


# ============================================================
# FINAL REPORT
# ============================================================

def final_dataset_report(source_counts, final_train_counts):
    path = os.path.join(REPORT_DIR, "final_dataset_report.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class", "original_count", "augmentation_level",
            "final_train_count", "valid_count", "test_count"
        ])

        for c in CLASSES:
            valid = len(get_image_files(
                os.path.join(OUTPUT_DIR, "valid", c)))
            test = len(get_image_files(
                os.path.join(OUTPUT_DIR, "test", c)))

            w.writerow([
                c,
                source_counts[c],
                get_augmentation_level(source_counts[c]),
                final_train_counts[c],
                valid,
                test,
            ])

    print(f"\nFinal report saved: {path}")
    print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("CAPSICUM DATASET PIPELINE - UPDATED")
    print("=" * 80)

    clean_generated_directories()
    validate_source_dataset()

    source_counts = count_source_dataset()
    print_augmentation_levels(source_counts)
    save_augmentation_report(source_counts)

    find_exact_duplicates()
    phash_groups = find_phash_groups()

    split_dataset(phash_groups)

    final_train_counts = augment_training_data(source_counts)

    copy_validation_test()

    final_dataset_report(
        source_counts,
        final_train_counts
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Training target: {TRAIN_TARGET} images/class")
    print(f"Expected train total: {EXPECTED_TRAIN_TOTAL}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
