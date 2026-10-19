"""
Plant Disease Detection - Training script
Transfer learning with MobileNetV2 on the PlantVillage dataset.

Folder structure expected:
    dataset/
        Apple___Apple_scab/
        Apple___healthy/
        Tomato___Late_blight/
        ...  (one folder per class, each with images)

Run:  python train.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- Config ----------------
DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 3       # train only the new top layers
EPOCHS_FINE = 2       # fine-tune last layers of MobileNetV2
SEED = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Data ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", num_classes)

# Save class names for the web app
with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------- Model ----------------
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------------- Train (stage 1: head only) ----------------
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# ---------------- Fine-tune (stage 2) ----------------
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

model.save(os.path.join(OUTPUT_DIR, "plant_model.keras"))

# ---------------- Plots ----------------
acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
val_loss = h1.history["val_loss"] + h2.history["val_loss"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# ---------------- Evaluation ----------------
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=90); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

final_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nFinal validation accuracy: {final_acc * 100:.2f}%")
print("Saved model, graphs and class names in the 'outputs' folder.")