import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cropnet_hybrid_model import build_cropnet_hybrid

# Paths
train_dir = "train"
test_dir = "test"
img_size = (224, 224)
batch_size = 32
num_classes = 38

# Data Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ----------------- Build & Compile Model -----------------
model = build_cropnet_hybrid(input_shape=(224, 224, 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------- Train Model -----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25
)

# ----------------- Evaluate on Test Set -----------------
test_loss, test_acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ----------------- Save Model -----------------
model.save("cropnet_hybrid_model_one.h5")
np.save("class_indices.npy", train_data.class_indices)

# ----------------- Plot Training & Validation Graph -----------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------- Plot Test Results (Line Chart) -----------------
plt.figure(figsize=(6, 5))
plt.plot([0, 1], [test_acc, test_loss], marker='o', linestyle='-', color='purple')
plt.xticks([0, 1], ["Test Accuracy", "Test Loss"])
plt.title("Test Accuracy vs Test Loss")
plt.ylabel("Value")
plt.grid(True)
plt.show()

