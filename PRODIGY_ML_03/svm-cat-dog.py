
    # ============================================================
# SVM CAT vs DOG IMAGE CLASSIFICATION
# Internship Project
# Dataset: Kaggle Dogs vs Cats
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Your current folder structure is:
#
# SVM_Cat_Dog/
# ├── train/
# │   └── train/
# │       ├── cat.0.jpg
# │       ├── cat.1.jpg
# │       └── ...
# └── svm-cat-dog.py

DATASET_PATH = "train/train"

# Resize every image
IMG_SIZE = 64

# Balanced dataset
CAT_LIMIT = 1000
DOG_LIMIT = 1000

# Train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Output files
MODEL_PATH = "svm_cat_dog_model.joblib"
RESULTS_DIR = "results"

# New image for prediction
TEST_IMAGE_PATH = "test.jpg"


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset folder not found: {DATASET_PATH}\n"
        "Make sure your images are inside train/train/"
    )


# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 65)
print("        SVM CAT VS DOG IMAGE CLASSIFICATION")
print("=" * 65)


# ============================================================
# 3. LOAD IMAGES
# ============================================================

images = []
labels = []

cat_count = 0
dog_count = 0

print("\n[1/9] Loading images...")

files = sorted(os.listdir(DATASET_PATH))

for filename in files:

    # -------------------------
    # CAT
    # -------------------------
    if filename.startswith("cat") and cat_count < CAT_LIMIT:
        label = 0
        cat_count += 1

    # -------------------------
    # DOG
    # -------------------------
    elif filename.startswith("dog") and dog_count < DOG_LIMIT:
        label = 1
        dog_count += 1

    else:
        continue

    image_path = os.path.join(
        DATASET_PATH,
        filename
    )

    try:

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Resize
        image = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        # Convert to NumPy array
        image_array = np.array(image)

        images.append(image_array)
        labels.append(label)

    except Exception as error:

        print(
            f"Warning: Could not load "
            f"{filename}: {error}"
        )

    # Stop after balanced dataset is reached
    if (
        cat_count == CAT_LIMIT
        and dog_count == DOG_LIMIT
    ):
        break


print("Image loading completed.")

print(f"  Cats : {cat_count}")
print(f"  Dogs : {dog_count}")
print(f"  Total: {len(images)}")


# ============================================================
# 4. CONVERT AND PREPROCESS DATA
# ============================================================

print("\n[2/9] Preparing image data...")

X = np.array(images)
y = np.array(labels)

print("Original shape:", X.shape)

# Flatten:
# 64 × 64 × 3 = 12,288 features

X = X.reshape(
    X.shape[0],
    -1
)

print(
    "Flattened shape:",
    X.shape
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[3/9] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 6. NORMALIZATION
# ============================================================

print("\n[4/9] Normalizing pixel values...")

# Original pixels: 0–255
# Normalized pixels: 0–1

X_train = (
    X_train.astype(np.float32)
    / 255.0
)

X_test = (
    X_test.astype(np.float32)
    / 255.0
)

print("Normalization completed.")


# ============================================================
# 7. CREATE AND TRAIN SVM
# ============================================================

print("\n[5/9] Training SVM model...")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

svm_model.fit(
    X_train,
    y_train
)

print("SVM training completed!")


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

dump(
    svm_model,
    MODEL_PATH
)

print(
    f"Saved model: {MODEL_PATH}"
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\n[6/9] Making predictions...")

y_pred = svm_model.predict(
    X_test
)

print("Predictions completed.")


# ============================================================
# 10. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 65)
print("                    RESULTS")
print("=" * 65)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy Percentage: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n[7/9] Generating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# Create figure
plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "SVM Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Cat", "Dog"]
)

plt.yticks(
    [0, 1],
    ["Cat", "Dog"]
)

# Put values inside cells
for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved: {confusion_path}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n[8/9] Generating classification report...")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Cat",
        "Dog"
    ]
)

print("\nClassification Report:")
print(report)

# Save report to file
report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "SVM Cat vs Dog Classification Report\n"
    )

    file.write(
        "=" * 45 + "\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(report)

print(
    f"Saved: {report_path}"
)


# ============================================================
# 13. SAMPLE PREDICTIONS
# ============================================================

print("\nGenerating sample predictions...")

plt.figure(
    figsize=(12, 9)
)

number_of_samples = min(
    9,
    len(X_test)
)

for i in range(
    number_of_samples
):

    # Convert flattened image back
    # into 64 × 64 × 3
    image = X_test[i].reshape(
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    # Actual label
    if y_test[i] == 0:
        actual = "Cat"
    else:
        actual = "Dog"

    # Predicted label
    if y_pred[i] == 0:
        predicted = "Cat"
    else:
        predicted = "Dog"

    plt.subplot(
        3,
        3,
        i + 1
    )

    plt.imshow(image)

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}"
    )

    plt.axis("off")


plt.tight_layout()

sample_path = os.path.join(
    RESULTS_DIR,
    "sample_predictions.png"
)

plt.savefig(
    sample_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved: {sample_path}"
)


# ============================================================
# 14. TEST A NEW IMAGE
# ============================================================

print("\n[9/9] Checking for new image...")

if os.path.exists(
    TEST_IMAGE_PATH
):

    print(
        f"Testing image: "
        f"{TEST_IMAGE_PATH}"
    )

    # Open image
    new_image = Image.open(
        TEST_IMAGE_PATH
    ).convert("RGB")

    # Resize
    new_image = new_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert to array
    new_image_array = np.array(
        new_image
    )

    # Flatten
    new_image_array = (
        new_image_array.reshape(
            1,
            -1
        )
    )

    # Normalize
    new_image_array = (
        new_image_array.astype(
            np.float32
        ) / 255.0
    )

    # Predict
    prediction = svm_model.predict(
        new_image_array
    )

    # Convert label
    if prediction[0] == 0:

        result = "Cat"

    else:

        result = "Dog"

    print(
        f"\nPrediction: {result}"
    )

    # Display image
    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        new_image
    )

    plt.title(
        f"SVM Prediction: {result}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

else:

    print(
        "\nNo test.jpg found."
    )

    print(
        "To test your own image, "
        "place a cat/dog image named "
        "'test.jpg' in the project folder."
    )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("                 PROJECT COMPLETED")
print("=" * 65)

print(
    f"\nTotal images used : {len(images)}"
)

print(
    f"Training images   : {len(X_train)}"
)

print(
    f"Testing images    : {len(X_test)}"
)

print(
    f"Image size        : "
    f"{IMG_SIZE} × {IMG_SIZE}"
)

print(
    "Classes           : Cat / Dog"
)

print(
    "Algorithm         : Support Vector Machine"
)

print(
    "Kernel            : RBF"
)

print(
    f"Accuracy           : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Model             : "
    f"{MODEL_PATH}"
)

print(
    f"Results           : "
    f"{RESULTS_DIR}/"
)

print("\nAll tasks completed successfully!")