import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import class_weight

# --- CONFIGURATION ---
DATA_DIR = "/app/hanabi_dataset_small"
OUTPUT_DIR = "/app" 
IMG_SIZE = (448, 448)
BATCH_SIZE = 8       
EPOCHS = 100         
LEARNING_RATE = 2e-5 

def run_training():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found. Run prepare_data.py first!")
        return

    # --- 1. LOAD DATA & GENERIC LABELS ---
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    
    # EXPORT LABELS.TXT IMMEDIATELY
    label_path = os.path.join(OUTPUT_DIR, "labels.txt")
    with open(label_path, "w") as f:
        for label in class_names:
            f.write(label + "\n")
    print(f"✅ Created {label_path} with {len(class_names)} classes.")

    # --- 2. CLASS WEIGHTS ---
    y_train = np.concatenate([y for x, y in train_ds], axis=0)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    # --- 3. BUILD MODEL (EfficientNetV2) ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2) 
    ])

    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(448, 448, 3), include_top=False, weights='imagenet')
    
    base_model.trainable = True
    for layer in base_model.layers[:50]: 
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=(448, 448, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5), 
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 4. CALLBACKS ---
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
    )

    # --- 5. TRAINING ---
    print(f"\nStarting High-Precision Training...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weight_dict, callbacks=[early_stop]
    )

    # --- 6. EXPORT TO TFLITE ---
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Basic optimization for mobile speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "hanabi_model_v2.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Save Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Final Model Performance')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_plot.png"))
    
    print(f"\n🎉 ALL DONE!")
    print(f"Files in your folder: hanabi_model_v2.tflite, labels.txt, accuracy_plot.png")

if __name__ == "__main__":
    run_training()