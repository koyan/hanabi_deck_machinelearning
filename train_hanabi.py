import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import class_weight

# --- CONFIGURATION ---
# We use the 'small' folder created by prepare_data.py
DATA_DIR = "/app/hanabi_dataset_small"
OUTPUT_DIR = "/app/output"
IMG_SIZE = (448, 448)  # High res to see Hanabi numbers clearly
BATCH_SIZE = 4
EPOCHS = 30

def run_training():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found. Did you run prepare_data.py first?")
        return

    # --- 1. DATASET SUMMARY ---
    print("-" * 30)
    print("HANABI TRAINING SESSION")
    categories = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(categories)} card categories.")
    
    # Quick count check
    for cat in categories:
        count = len(os.listdir(os.path.join(DATA_DIR, cat)))
        if count < 5:
            print(f" ! WARNING: {cat} only has {count} images. Accuracy may suffer.")
    print("-" * 30)

    # --- 2. LOAD DATA ---
    print("Loading images into memory...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names

    # --- 3. CALCULATE CLASS WEIGHTS ---
    # This prevents cards with 35 photos from 'bullying' cards with 11 photos
    print("Balancing dataset weights...")
    y_train = np.concatenate([y for x, y in train_ds], axis=0)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train)
    class_weight_dict = dict(enumerate(weights))

    # --- 4. BUILD MODEL ---
    print("Assembling Neural Network (MobileNetV2 Fine-Tuning)...")
    
    # Data Augmentation (Artificial variety)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1)
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(448, 448, 3), include_top=False, weights='imagenet')
    
    # Unfreeze the top layers for better card-specific recognition
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=(448, 448, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 5. TRAINING ---
    print("\nStarting Training. Grab a coffee, this might take a while...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict
    )

    # --- 6. EXPORT TO TFLITE ---
    print("\nTraining complete. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_filename = os.path.join(OUTPUT_DIR, "hanabi_model_v1.tflite")
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    print(f"SUCCESS: Model saved to {tflite_filename}")

    # --- 7. SAVE ACCURACY PLOT ---
    print("Saving accuracy plot to output folder...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Hanabi Model Performance')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_plot.png"))
    print("Plot saved as 'accuracy_plot.png'.")

if __name__ == "__main__":
    run_training()