
import os
import random
import json
import time
import copy

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "train"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "valid"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "training_results"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASS_NAMES = [

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

    "virus"
]


NUM_CLASSES = len(
    CLASS_NAMES
)

TRAIN_TARGET = 1100

EXPECTED_TRAIN_TOTAL = (
    NUM_CLASSES
    * TRAIN_TARGET
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

PATIENCE = 3

MIN_DELTA = 0.001

GRADIENT_CLIP = 1.0

SEED = 42


# ============================================================
# OFFLINE AUGMENTATION CONFIGURATION
# ============================================================
#
# These settings describe augmentation.py.
# Actual augmentation is NOT performed here.
# ============================================================

AUGMENTATION_CONFIG = {

    "method": "Offline augmentation",

    "source": "augmentation.py",

    "train_only": True,

    "validation_augmented": False,

    "test_augmented": False,

    "target_per_class": TRAIN_TARGET,

    "levels": {

        "strong": {
            "condition": "< 300"
        },

        "moderate_strong": {
            "condition": "300 - 500"
        },

        "moderate": {
            "condition": "501 - 800"
        },

        "light": {
            "condition": "801 - 1000"
        },

        "very_light": {
            "condition": "1001 - 1100"
        },

        "none": {
            "condition": "> 1100"
        }

    }

}


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = [

    0.485,

    0.456,

    0.406
]


IMAGENET_STD = [

    0.229,

    0.224,

    0.225
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# CUDA SETTINGS
# ============================================================

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


# ============================================================
# START MESSAGE
# ============================================================

print()

print("=" * 80)

print(
    "CAPSICUM RESNET50 - IMPROVED FINAL TRAINING"
)

print("=" * 80)

print()

print(
    "Device:",
    device
)

print(
    "Number of classes:",
    NUM_CLASSES
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "Dropout:",
    DROPOUT
)

print(
    "Gradient clipping:",
    GRADIENT_CLIP
)

print()

print(
    "Offline augmentation:"
)

print(
    "  <300       -> STRONG"
)

print(
    "  300-500    -> MODERATE-STRONG"
)

print(
    "  501-800    -> MODERATE"
)

print(
    "  801-1000   -> LIGHT"
)

print(
    "  1001-1100  -> VERY-LIGHT"
)

print(
    "  >1100      -> NONE"
)

print()


# ============================================================
# PATH CHECK
# ============================================================

print("=" * 80)

print(
    "CHECKING DATASET PATHS"
)

print("=" * 80)

print()

print(
    "Train directory:"
)

print(
    TRAIN_DIR
)

print()

print(
    "Validation directory:"
)

print(
    VALID_DIR
)

print()

print(
    "Test directory:"
)

print(
    TEST_DIR
)

print()


if not os.path.isdir(
    TRAIN_DIR
):

    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )


if not os.path.isdir(
    VALID_DIR
):

    raise FileNotFoundError(
        f"Validation directory not found:\n{VALID_DIR}"
    )


if not os.path.isdir(
    TEST_DIR
):

    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# TRANSFORMS
# ============================================================
#
# IMPORTANT:
#
# augmentation.py has already performed offline augmentation.
#
# Therefore:
#
# TRAIN -> Resize + Normalize
# VALID -> Resize + Normalize
# TEST  -> Resize + Normalize
#
# No random augmentation here.
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


valid_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 80)

print(
    "LOADING DATASETS"
)

print("=" * 80)

print()


train_dataset = datasets.ImageFolder(

    TRAIN_DIR,

    transform=train_transform
)


valid_dataset = datasets.ImageFolder(

    VALID_DIR,

    transform=valid_transform
)


test_dataset = datasets.ImageFolder(

    TEST_DIR,

    transform=test_transform
)


# ============================================================
# CLASS ORDER CHECK
# ============================================================

print(
    "Detected train classes:"
)

print(
    train_dataset.classes
)

print()

print(
    "Detected validation classes:"
)

print(
    valid_dataset.classes
)

print()

print(
    "Detected test classes:"
)

print(
    test_dataset.classes
)

print()


if train_dataset.classes != CLASS_NAMES:

    raise ValueError(

        "\nTrain class order mismatch!\n\n"

        f"Expected:\n{CLASS_NAMES}\n\n"

        f"Found:\n{train_dataset.classes}"
    )


if valid_dataset.classes != CLASS_NAMES:

    raise ValueError(

        "\nValidation class order mismatch!\n\n"

        f"Expected:\n{CLASS_NAMES}\n\n"

        f"Found:\n{valid_dataset.classes}"
    )


if test_dataset.classes != CLASS_NAMES:

    raise ValueError(

        "\nTest class order mismatch!\n\n"

        f"Expected:\n{CLASS_NAMES}\n\n"

        f"Found:\n{test_dataset.classes}"
    )


# ============================================================
# DATASET COUNT FUNCTION
# ============================================================

def count_dataset_classes(
    dataset
):

    counts = {

        class_name: 0

        for class_name in CLASS_NAMES
    }


    for _, label in dataset.samples:

        class_name = (
            dataset.classes[label]
        )

        counts[class_name] += 1


    return counts


# ============================================================
# DATASET COUNTS
# ============================================================

train_counts = count_dataset_classes(
    train_dataset
)

valid_counts = count_dataset_classes(
    valid_dataset
)

test_counts = count_dataset_classes(
    test_dataset
)


# ============================================================
# PRINT TRAIN COUNTS
# ============================================================

print("=" * 80)

print(
    "TRAIN DATASET COUNTS"
)

print("=" * 80)

print()


for class_name in CLASS_NAMES:

    print(

        f"{class_name:<25} : "
        f"{train_counts[class_name]}"
    )


print()

print(
    "Total training images:",
    len(train_dataset)
)

print(
    "Expected training images:",
    EXPECTED_TRAIN_TOTAL
)

print()


# ============================================================
# EXACT TRAIN DATASET VALIDATION
# ============================================================

if len(train_dataset) != EXPECTED_TRAIN_TOTAL:

    raise ValueError(

        "\nTraining dataset size is incorrect!\n"

        f"Expected: {EXPECTED_TRAIN_TOTAL}\n"

        f"Found: {len(train_dataset)}\n\n"

        "Run augmentation.py again."
    )


for class_name in CLASS_NAMES:

    count = train_counts[
        class_name
    ]


    if count != TRAIN_TARGET:

        raise ValueError(

            f"\nIncorrect train count "
            f"for {class_name}\n"

            f"Expected: {TRAIN_TARGET}\n"

            f"Found: {count}"
        )


print(
    "SUCCESS: Every training class "
    f"contains exactly {TRAIN_TARGET} images."
)

print()


# ============================================================
# VALIDATION COUNTS
# ============================================================

print("=" * 80)

print(
    "VALIDATION COUNTS"
)

print("=" * 80)

print()


for class_name in CLASS_NAMES:

    print(

        f"{class_name:<25} : "
        f"{valid_counts[class_name]}"
    )


print()

print(
    "Total validation images:",
    len(valid_dataset)
)

print()


# ============================================================
# TEST COUNTS
# ============================================================

print("=" * 80)

print(
    "TEST COUNTS"
)

print("=" * 80)

print()


for class_name in CLASS_NAMES:

    print(

        f"{class_name:<25} : "
        f"{test_counts[class_name]}"
    )


print()

print(
    "Total test images:",
    len(test_dataset)
)

print()


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# DATA LOADER INFORMATION
# ============================================================

print("=" * 80)

print(
    "DATA LOADERS READY"
)

print("=" * 80)

print()

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(valid_loader)
)

print(
    "Test batches:",
    len(test_loader)
)

print()


# ============================================================
# LOAD RESNET50
# ============================================================

print("=" * 80)

print(
    "LOADING RESNET50"
)

print("=" * 80)

print()


pretrained_used = False


try:

    weights = (
        models.ResNet50_Weights.DEFAULT
    )

    model = models.resnet50(
        weights=weights
    )

    pretrained_used = True

    print(
        "ImageNet pretrained weights loaded."
    )


except Exception as e:

    print(
        "WARNING: Could not load "
        "ImageNet pretrained weights."
    )

    print(
        "Reason:",
        e
    )

    print()

    print(
        "Using ResNet50 without pretrained weights."
    )

    model = models.resnet50(
        weights=None
    )


# ============================================================
# FREEZE EVERYTHING
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# UNFREEZE LAYER4
# ============================================================

for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# REPLACE FC
# ============================================================

model.fc = nn.Sequential(

    nn.Dropout(
        p=DROPOUT
    ),

    nn.Linear(
        2048,
        NUM_CLASSES
    )
)


# FC automatically trainable,
# but explicitly confirm.

for param in model.fc.parameters():

    param.requires_grad = True


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    device
)


# ============================================================
# FROZEN BATCHNORM HELPER
# ============================================================

def freeze_frozen_batchnorm(
    model
):

    for module in model.layer1.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


    for module in model.layer2.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


    for module in model.layer3.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

print("=" * 80)

print(
    "FINE-TUNING CONFIGURATION"
)

print("=" * 80)

print()

print(
    "Layer1 : FROZEN"
)

print(
    "Layer2 : FROZEN"
)

print(
    "Layer3 : FROZEN"
)

print(
    "Layer4 : TRAINABLE"
)

print(
    "FC     : TRAINABLE"
)

print()

print(
    "FC:"
)

print(
    model.fc
)

print()


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = 0

trainable_params = 0

frozen_params = 0


for param in model.parameters():

    parameter_count = (
        param.numel()
    )

    total_params += (
        parameter_count
    )


    if param.requires_grad:

        trainable_params += (
            parameter_count
        )

    else:

        frozen_params += (
            parameter_count
        )


print("=" * 80)

print(
    "MODEL PARAMETERS"
)

print("=" * 80)

print()

print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print(
    f"Frozen parameters    : "
    f"{frozen_params:,}"
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    filter(

        lambda parameter:
        parameter.requires_grad,

        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=1
)


# ============================================================
# AMP
# ============================================================

use_amp = (
    device.type == "cuda"
)


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(

    model,

    loader,

    criterion,

    optimizer,

    device,

    scaler

):

    model.train()

    freeze_frozen_batchnorm(
        model
    )


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(

            device,

            non_blocking=True
        )

        labels = labels.to(

            device,

            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # CUDA AMP
        # ----------------------------------------------------

        if scaler is not None:

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16

            ):

                outputs = model(
                    images
                )

                loss = criterion(

                    outputs,

                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=GRADIENT_CLIP
            )


            scaler.step(
                optimizer
            )

            scaler.update()


        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        else:

            outputs = model(
                images
            )

            loss = criterion(

                outputs,

                labels
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=GRADIENT_CLIP
            )


            optimizer.step()


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        running_loss += (

            loss.item()
            * images.size(0)
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        _, predicted = torch.max(

            outputs,

            1
        )


        total += (
            labels.size(0)
        )


        correct += (

            predicted == labels

        ).sum().item()


    epoch_loss = (

        running_loss
        / total
    )


    epoch_accuracy = (

        correct
        / total
    ) * 100


    return (

        epoch_loss,

        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_one_epoch(

    model,

    loader,

    criterion,

    device

):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(

                device,

                non_blocking=True
            )

            labels = labels.to(

                device,

                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(

                outputs,

                labels
            )


            running_loss += (

                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(

                outputs,

                1
            )


            total += (
                labels.size(0)
            )


            correct += (

                predicted == labels

            ).sum().item()


    epoch_loss = (

        running_loss
        / total
    )


    epoch_accuracy = (

        correct
        / total
    ) * 100


    return (

        epoch_loss,

        epoch_accuracy
    )


# ============================================================
# HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "valid_loss": [],

    "valid_accuracy": [],

    "learning_rate": []
}


# ============================================================
# BEST MODEL VARIABLES
# ============================================================

best_val_loss = float(
    "inf"
)

best_val_accuracy = 0.0

best_epoch = 0

best_model_state = None

patience_counter = 0


# ============================================================
# TRAINING START
# ============================================================

print()

print("=" * 80)

print(
    "TRAINING STARTED"
)

print("=" * 80)

print()


training_start_time = (
    time.time()
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    EPOCHS
):

    epoch_start_time = (
        time.time()
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(

        model,

        train_loader,

        criterion,

        optimizer,

        device,

        scaler
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_loss, valid_accuracy = validate_one_epoch(

        model,

        valid_loader,

        criterion,

        device
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        valid_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )

    history[
        "valid_loss"
    ].append(
        valid_loss
    )

    history[
        "valid_accuracy"
    ].append(
        valid_accuracy
    )

    history[
        "learning_rate"
    ].append(
        current_lr
    )


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_time = (

        time.time()
        - epoch_start_time
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("=" * 80)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print("=" * 80)

    print()

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Valid Loss      : "
        f"{valid_loss:.4f}"
    )

    print(
        f"Valid Accuracy  : "
        f"{valid_accuracy:.2f}%"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.8f}"
    )

    print(
        f"Time            : "
        f"{epoch_time:.2f} sec"
    )

    print()


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (

        valid_loss
        < best_val_loss - MIN_DELTA

    ):

        best_val_loss = (
            valid_loss
        )

        best_val_accuracy = (
            valid_accuracy
        )

        best_epoch = (
            epoch + 1
        )

        best_model_state = copy.deepcopy(

            model.state_dict()
        )

        patience_counter = 0


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        best_model_path = os.path.join(

            MODEL_DIR,

            "condition_resnet50_best.pth"
        )


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "num_classes":
                    NUM_CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "model_name":
                    "ResNet50",

                "pretrained":
                    pretrained_used,

                "pretrained_source":
                    "ImageNet"
                    if pretrained_used
                    else None,

                "fine_tuning":
                    "Layer4 + FC",

                "train_target":
                    TRAIN_TARGET,

                "expected_train_total":
                    EXPECTED_TRAIN_TOTAL,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss,

                "best_val_accuracy":
                    best_val_accuracy,

                "seed":
                    SEED,

                "augmentation":
                    AUGMENTATION_CONFIG,

                "normalization":
                    {

                        "mean":
                            IMAGENET_MEAN,

                        "std":
                            IMAGENET_STD

                    }

            },

            best_model_path
        )


        print(
            ">>> BEST MODEL SAVED"
        )

        print(
            best_model_path
        )

        print()


    else:

        patience_counter += 1


        print(
            "No significant validation "
            "loss improvement."
        )

        print(

            f"Early stopping counter: "
            f"{patience_counter}/{PATIENCE}"
        )

        print()


        if (

            patience_counter
            >= PATIENCE

        ):

            print("=" * 80)

            print(
                "EARLY STOPPING"
            )

            print("=" * 80)

            print()

            break


# ============================================================
# TRAINING FINISHED
# ============================================================

training_time = (

    time.time()
    - training_start_time
)


print()

print("=" * 80)

print(
    "TRAINING COMPLETED"
)

print("=" * 80)

print()

print(

    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

print()

print(
    "Best epoch:",
    best_epoch
)

print(

    "Best validation loss:",

    f"{best_val_loss:.4f}"
)

print(

    "Best validation accuracy:",

    f"{best_val_accuracy:.2f}%"
)

print()


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(

        best_model_state
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "condition_resnet50.pth"
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "class_to_idx":
            train_dataset.class_to_idx,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "model_name":
            "ResNet50",

        "pretrained":
            pretrained_used,

        "pretrained_source":
            "ImageNet"
            if pretrained_used
            else None,

        "fine_tuning":
            "Layer4 + FC",

        "train_target":
            TRAIN_TARGET,

        "expected_train_total":
            EXPECTED_TRAIN_TOTAL,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "best_val_accuracy":
            best_val_accuracy,

        "training_config":
            {

                "batch_size":
                    BATCH_SIZE,

                "epochs":
                    EPOCHS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dropout":
                    DROPOUT,

                "patience":
                    PATIENCE,

                "min_delta":
                    MIN_DELTA,

                "gradient_clip":
                    GRADIENT_CLIP,

                "seed":
                    SEED,

                "image_size":
                    IMAGE_SIZE

            },

        "augmentation":
            AUGMENTATION_CONFIG,

        "normalization":
            {

                "mean":
                    IMAGENET_MEAN,

                "std":
                    IMAGENET_STD

            }

    },

    final_model_path
)


print("=" * 80)

print(
    "FINAL MODEL SAVED"
)

print("=" * 80)

print()

print(
    final_model_path
)

print()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("=" * 80)

print(
    "FINAL INTERNAL TEST"
)

print("=" * 80)

print()


model.eval()


all_predictions = []

all_labels = []


test_loss_total = 0.0

test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(

            device,

            non_blocking=True
        )

        labels = labels.to(

            device,

            non_blocking=True
        )


        outputs = model(
            images
        )


        loss = criterion(

            outputs,

            labels
        )


        test_loss_total += (

            loss.item()
            * images.size(0)
        )


        _, predictions = torch.max(

            outputs,

            1
        )


        all_predictions.extend(

            predictions.cpu().numpy()
        )


        all_labels.extend(

            labels.cpu().numpy()
        )


        test_total += (
            labels.size(0)
        )


# ============================================================
# TEST LOSS
# ============================================================

test_loss = (

    test_loss_total
    / test_total
)


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = (

    accuracy_score(

        all_labels,

        all_predictions

    )

    * 100
)


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print(

    "Test Loss:",

    f"{test_loss:.4f}"
)

print()

print(

    "Test Accuracy:",

    f"{test_accuracy:.2f}%"
)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_labels,

    all_predictions,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)


print("=" * 80)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 80)

print()

print(
    report
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(

    RESULT_DIR,

    "classification_report.txt"
)


with open(

    report_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        "CAPSICUM RESNET50 - FINAL INTERNAL TEST\n"
    )

    file.write(

        "=" * 80
        + "\n\n"
    )

    file.write(

        f"Test Loss: "
        f"{test_loss:.4f}\n"
    )

    file.write(

        f"Test Accuracy: "
        f"{test_accuracy:.2f}%\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    )
)


# ============================================================
# SAVE CONFUSION MATRIX JSON
# ============================================================

cm_path = os.path.join(

    RESULT_DIR,

    "confusion_matrix.json"
)


with open(

    cm_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        cm.tolist(),

        file,

        indent=4
    )


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(

    figsize=(14, 12)
)


plt.imshow(

    cm,

    interpolation="nearest"
)


plt.title(

    "Capsicum ResNet50 - Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(

    NUM_CLASSES
)


plt.xticks(

    tick_marks,

    CLASS_NAMES,

    rotation=90
)


plt.yticks(

    tick_marks,

    CLASS_NAMES
)


threshold = (

    cm.max()
    / 2.0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            str(
                cm[i, j]
            ),

            horizontalalignment="center",

            verticalalignment="center",

            color=(

                "white"

                if cm[i, j] > threshold

                else "black"
            )
        )


plt.ylabel(
    "True Label"
)


plt.xlabel(
    "Predicted Label"
)


plt.tight_layout()


confusion_plot_path = os.path.join(

    RESULT_DIR,

    "confusion_matrix.png"
)


plt.savefig(

    confusion_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

epochs_range = range(

    1,

    len(
        history["train_loss"]
    ) + 1
)


plt.figure(

    figsize=(10, 6)
)


plt.plot(

    epochs_range,

    history["train_loss"],

    label="Train Loss"
)


plt.plot(

    epochs_range,

    history["valid_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(

    "ResNet50 Training and Validation Loss"
)


plt.legend()


plt.grid(

    True,

    alpha=0.3
)


plt.tight_layout()


loss_plot_path = os.path.join(

    RESULT_DIR,

    "loss_curve.png"
)


plt.savefig(

    loss_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(

    figsize=(10, 6)
)


plt.plot(

    epochs_range,

    history["train_accuracy"],

    label="Train Accuracy"
)


plt.plot(

    epochs_range,

    history["valid_accuracy"],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(

    "ResNet50 Training and Validation Accuracy"
)


plt.legend()


plt.grid(

    True,

    alpha=0.3
)


plt.tight_layout()


accuracy_plot_path = os.path.join(

    RESULT_DIR,

    "accuracy_curve.png"
)


plt.savefig(

    accuracy_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(

    RESULT_DIR,

    "training_history.json"
)


with open(

    history_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        history,

        file,

        indent=4
    )


# ============================================================
# SAVE DATASET COUNTS
# ============================================================

dataset_counts = {

    "classes":
        CLASS_NAMES,

    "num_classes":
        NUM_CLASSES,

    "train_target_per_class":
        TRAIN_TARGET,

    "expected_train_total":
        EXPECTED_TRAIN_TOTAL,

    "train_total":
        len(train_dataset),

    "valid_total":
        len(valid_dataset),

    "test_total":
        len(test_dataset),

    "train":
        train_counts,

    "valid":
        valid_counts,

    "test":
        test_counts

}


dataset_counts_path = os.path.join(

    RESULT_DIR,

    "dataset_counts.json"
)


with open(

    dataset_counts_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        dataset_counts,

        file,

        indent=4
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

training_summary = {

    "project":
        "Capsicum Plant Condition Classification",

    "model":
        "ResNet50",

    "pretrained":
        pretrained_used,

    "pretrained_source":
        "ImageNet"
        if pretrained_used
        else None,

    "fine_tuning":
        {

            "layer1":
                "Frozen",

            "layer2":
                "Frozen",

            "layer3":
                "Frozen",

            "layer4":
                "Trainable",

            "fc":
                "Trainable"

        },

    "fc_architecture":
        "Dropout(0.30) + Linear(2048 -> 13)",

    "num_classes":
        NUM_CLASSES,

    "class_names":
        CLASS_NAMES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(
            history["train_loss"]
        ),

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "dropout":
        DROPOUT,

    "gradient_clip":
        GRADIENT_CLIP,

    "train_target_per_class":
        TRAIN_TARGET,

    "train_total":
        len(train_dataset),

    "validation_total":
        len(valid_dataset),

    "test_total":
        len(test_dataset),

    "best_epoch":
        best_epoch,

    "best_validation_loss":
        best_val_loss,

    "best_validation_accuracy":
        best_val_accuracy,

    "test_loss":
        test_loss,

    "test_accuracy":
        test_accuracy,

    "training_time_minutes":
        training_time / 60,

    "augmentation":
        AUGMENTATION_CONFIG,

    "normalization":
        {

            "mean":
                IMAGENET_MEAN,

            "std":
                IMAGENET_STD

        }

}


summary_path = os.path.join(

    RESULT_DIR,

    "training_summary.json"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        training_summary,

        file,

        indent=4
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(

    MODEL_DIR,

    "condition_resnet50_checkpoint.pth"
)


torch.save(

    {

        "epoch":
            len(
                history["train_loss"]
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "class_names":
            CLASS_NAMES,

        "class_to_idx":
            train_dataset.class_to_idx,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "pretrained":
            pretrained_used,

        "pretrained_source":
            "ImageNet"
            if pretrained_used
            else None,

        "fine_tuning":
            "Layer4 + FC",

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "best_val_accuracy":
            best_val_accuracy,

        "augmentation":
            AUGMENTATION_CONFIG,

        "history":
            history

    },

    checkpoint_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 80)

print(
    "ALL TRAINING OUTPUTS SAVED"
)

print("=" * 80)

print()

print(
    "MODEL FILES:"
)

print(

    "1.",

    os.path.join(

        MODEL_DIR,

        "condition_resnet50_best.pth"
    )
)

print(

    "2.",

    final_model_path
)

print(

    "3.",

    checkpoint_path
)

print()

print(
    "RESULT FILES:"
)

print(
    "1.",
    report_path
)

print(
    "2.",
    cm_path
)

print(
    "3.",
    confusion_plot_path
)

print(
    "4.",
    loss_plot_path
)

print(
    "5.",
    accuracy_plot_path
)

print(
    "6.",
    history_path
)

print(
    "7.",
    dataset_counts_path
)

print(
    "8.",
    summary_path
)

print()

print("=" * 80)

print(
    "FINAL RESULTS"
)

print("=" * 80)

print()

print(

    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(

    f"Final Test Accuracy      : "
    f"{test_accuracy:.2f}%"
)

print(

    f"Final Test Loss          : "
    f"{test_loss:.4f}"
)

print()

print(
    "Training configuration:"
)

print(

    "ResNet50 ImageNet Pretrained"
    if pretrained_used
    else
    "ResNet50 without pretrained weights"
)

print(
    "Layer1: Frozen"
)

print(
    "Layer2: Frozen"
)

print(
    "Layer3: Frozen"
)

print(
    "Layer4: Trainable"
)

print(
    "FC: Trainable"
)

print(
    "FC: Dropout(0.30) + Linear(2048 -> 13)"
)

print(
    "Gradient clipping:",
    GRADIENT_CLIP
)

print()

print(
    "Offline augmentation:"
)

print(
    "Strong / Moderate-Strong / Moderate / "
    "Light / Very-Light / None"
)

print(
    "Train target:",
    TRAIN_TARGET,
    "images/class"
)

print(
    "Total training images:",
    EXPECTED_TRAIN_TOTAL
)

print()

print("=" * 80)

print(
    "CAPSICUM RESNET50 TRAINING FINISHED"
)

print("=" * 80)

print()

