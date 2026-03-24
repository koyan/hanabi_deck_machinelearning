import tensorflow as tf
import numpy as np
import os
from collections import defaultdict

# --- CONFIG ---
DATA_DIR = "/app/hanabi_dataset_small"
MODEL_PATH = "/app/hanabi_model_v1.tflite"
IMG_SIZE = (448, 448)

def generate_error_report():
    print("--- Starting Global Validation Audit ---")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123, 
        image_size=IMG_SIZE,
        batch_size=1
    )

    class_names = val_ds.class_names

    # Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # stats: { "card_name": [total_seen, wrong_guesses, { "confused_with": count }] }
    stats = defaultdict(lambda: [0, 0, defaultdict(int)])
    total_correct = 0
    total_images = 0

    print("Analyzing every image...")
    for images, labels in val_ds:
        img = images[0].numpy()
        true_label = class_names[labels[0]]

        # Run TFLite Inference
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_idx = np.argmax(output_data[0])
        pred_label = class_names[pred_idx]

        # Update Global Stats
        total_images += 1
        stats[true_label][0] += 1
        
        if pred_label == true_label:
            total_correct += 1
        else:
            stats[true_label][1] += 1
            stats[true_label][2][pred_label] += 1

    # --- GLOBAL SCORECARD ---
    global_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "#" * 45)
    print(f" GLOBAL VALIDATION SCORE: {total_correct} / {total_images}")
    print(f" TOTAL ACCURACY:           {global_accuracy:.2f}%")
    print("#" * 45)

    # --- PRINTOUT 1: ACTION PLAN (ERROR FOCUS) ---
    print("\n" + "!" * 45)
    print(f"{'TOP FAILURES':<25} | {'ERRORS/TOTAL':<12} | {'FAILURE %'}")
    print("-" * 45)

    error_only_stats = [item for item in stats.items() if item[1][1] > 0]
    sorted_by_fail = sorted(error_only_stats, key=lambda x: (x[1][1]/x[1][0]), reverse=True)

    for card_name, counts in sorted_by_fail:
        total, errors, confusions = counts
        fail_percent = (errors / total * 100)
        most_common_mistake = max(confusions, key=confusions.get) if confusions else "N/A"
        print(f"{card_name:<25} | {errors:>2}/{total:<2}         | {fail_percent:>5.1f}%")
        print(f"   ↳ Most confused with: {most_common_mistake}")

    # --- PRINTOUT 2: FULL INVENTORY (ALPHABETICAL) ---
    print("\n" + "=" * 45)
    print(f"{'FULL CARD INVENTORY':<25} | {'SUCCESS RATE':<12} | {'STATUS'}")
    print("-" * 45)

    all_cards_sorted = sorted(stats.items())

    for card_name, counts in all_cards_sorted:
        total, errors, _ = counts
        success = total - errors
        success_percent = (success / total * 100) if total > 0 else 0
        
        if success_percent == 100: status = "✅ PERFECT"
        elif success_percent >= 80: status = "🟢 GOOD"
        elif success_percent >= 50: status = "⚠️ POOR"
        else: status = "❌ CRITICAL"

        print(f"{card_name:<25} | {success:>2}/{total:<2}         | {status}")

    print("=" * 45)

if __name__ == "__main__":
    generate_error_report()