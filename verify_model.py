import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
DATA_DIR = "/app/hanabi_dataset_small"
MODEL_PATH = "/app/hanabi_model_v1.tflite"
IMG_SIZE = (448, 448)

def verify_on_validation_set():
    # 1. Load the "Validation" portion (Must match training seed)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123, 
        image_size=IMG_SIZE,
        batch_size=9 
    )

    class_names = val_ds.class_names

    # 2. Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Predict and Count
    correct_count = 0
    total_to_test = 9

    for images, labels in val_ds.take(1):
        plt.figure(figsize=(15, 15))
        
        for i in range(total_to_test):
            img = images[i].numpy()
            true_label = class_names[labels[i]]

            # Prep for TFLite
            input_data = np.expand_dims(img, axis=0).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = np.argmax(output_data[0])
            pred_label = class_names[pred_idx]
            confidence = output_data[0][pred_idx] * 100

            # Update score
            if pred_label == true_label:
                correct_count += 1
                color = "green"
            else:
                color = "red"

            # Plot
            plt.subplot(3, 3, i + 1)
            plt.imshow(img.astype("uint8"))
            plt.title(f"Actual: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", color=color)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("/app/validation_results.png")
        
        print("\n" + "="*30)
        print(f"VERIFICATION SCORE: {correct_count} / {total_to_test}")
        print(f"Percentage: {(correct_count/total_to_test)*100:.1f}%")
        print("="*30)
        print("Detailed results saved to: /app/validation_results.png")

if __name__ == "__main__":
    verify_on_validation_set()