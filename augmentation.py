import os


# ============================================================
# CAPSICUM DATASET IMAGE COUNT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "data new"
)


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# COUNT
# ============================================================

total_images = 0
total_classes = 0


print("\n" + "=" * 70)
print("CAPSICUM DATASET IMAGE COUNT")
print("=" * 70)

print("\nDataset Location:")
print(DATASET_DIR)


# ============================================================
# CHECK FOLDER
# ============================================================

if not os.path.isdir(DATASET_DIR):

    print("\nERROR: data new folder not found.")
    raise SystemExit


# ============================================================
# CLASS-WISE COUNT
# ============================================================

print("\n" + "-" * 70)
print("CLASS-WISE IMAGE COUNT")
print("-" * 70)


class_counts = {}


for class_name in sorted(os.listdir(DATASET_DIR)):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    count = 0

    for root, dirs, files in os.walk(class_path):

        for file_name in files:

            if file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                count += 1

    class_counts[class_name] = count

    total_classes += 1
    total_images += count

    print(
        f"{class_name:<35} : {count:>5}"
    )


# ============================================================
# TOTAL
# ============================================================

print("\n" + "=" * 70)
print("TOTAL")
print("=" * 70)

print(
    f"{'Total Classes':<35} : {total_classes}"
)

print(
    f"{'Total Images':<35} : {total_images}"
)

print("=" * 70)

print("\nCOUNTING COMPLETED")