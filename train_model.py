"""
Train a handwriting recognition model on EMNIST (letters + digits).
Run this ONCE before using the air writing app.

Supports two backends:
  - TensorFlow/Keras CNN (best accuracy, ~86%)
  - scikit-learn MLP (lighter, ~78%, no GPU needed)

Usage:
  python train_model.py              # Auto-detect best backend
  python train_model.py --sklearn    # Force scikit-learn
  python train_model.py --tensorflow # Force TensorFlow
"""

import os
import sys
import numpy as np
import struct
import gzip
import urllib.request
import pickle

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# EMNIST ByClass dataset URLs
EMNIST_BASE = "https://biometrics.nist.gov/cs_links/EMNIST/gzip/"
EMNIST_FILES = {
    "train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-byclass-test-labels-idx1-ubyte.gz",
}

NUM_CLASSES = 62  # 10 digits + 26 uppercase + 26 lowercase


def label_to_char(label):
    """Convert EMNIST label to character."""
    if label < 10:
        return chr(ord('0') + label)
    elif label < 36:
        return chr(ord('A') + label - 10)
    else:
        return chr(ord('a') + label - 36)


def download_emnist():
    """Download EMNIST dataset files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, filename in EMNIST_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✓ Already downloaded: {filename}")
            continue
        url = EMNIST_BASE + filename
        print(f"  ↓ Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Done: {filename}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            raise


def load_emnist_images(filepath):
    """Load EMNIST images from gzipped IDX file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        # EMNIST images are stored transposed
        data = np.transpose(data, (0, 2, 1))
    return data


def load_emnist_labels(filepath):
    """Load EMNIST labels from gzipped IDX file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def save_class_mapping():
    """Save the label-to-character mapping."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_map = np.array([label_to_char(i) for i in range(NUM_CLASSES)])
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.npy")
    np.save(mapping_path, class_map)
    print(f"  ✓ Class mapping saved: {mapping_path}")
    return class_map


def train_sklearn(train_images, train_labels, test_images, test_labels):
    """Train using scikit-learn MLP classifier."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n  Using scikit-learn MLP classifier")
    print("  (This may take 5-10 minutes on CPU...)\n")

    # Flatten images: (N, 28, 28) → (N, 784)
    X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    # Use a subset for faster training (full EMNIST is huge)
    # Take max 100k samples for training
    max_train = min(100_000, len(X_train))
    max_test = min(20_000, len(X_test))

    indices = np.random.RandomState(42).permutation(len(X_train))[:max_train]
    X_train = X_train[indices]
    y_train = train_labels[indices]

    indices = np.random.RandomState(42).permutation(len(X_test))[:max_test]
    X_test = X_test[indices]
    y_test = test_labels[indices]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train MLP
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        batch_size=256,
        max_iter=30,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"\n  Test accuracy: {accuracy:.4f}")

    # Save model + scaler
    model_path = os.path.join(MODEL_DIR, "emnist_sklearn.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"  ✓ Model saved: {model_path}")

    return accuracy


def train_tensorflow(train_images, train_labels, test_images, test_labels):
    """Train using TensorFlow/Keras CNN."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\n  Using TensorFlow/Keras CNN")

    # Normalize and reshape
    X_train = train_images.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
    X_test = test_images.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0

    # Build CNN
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model.fit(X_train, train_labels,
              epochs=5, batch_size=128,
              validation_split=0.1, verbose=1)

    test_loss, accuracy = model.evaluate(X_test, test_labels, verbose=0)
    print(f"\n  Test accuracy: {accuracy:.4f}")

    model_path = os.path.join(MODEL_DIR, "emnist_model.keras")
    model.save(model_path)
    print(f"  ✓ Model saved: {model_path}")

    return accuracy


def main():
    print("=" * 55)
    print("   EMNIST Handwriting Model Trainer")
    print("=" * 55)

    # Determine backend
    use_sklearn = "--sklearn" in sys.argv
    use_tf = "--tensorflow" in sys.argv

    if not use_sklearn and not use_tf:
        # Auto-detect
        try:
            import tensorflow
            use_tf = True
            print(f"\n  Auto-detected: TensorFlow {tensorflow.__version__}")
        except ImportError:
            try:
                import sklearn
                use_sklearn = True
                print(f"\n  Auto-detected: scikit-learn {sklearn.__version__}")
            except ImportError:
                print("\n  ERROR: Neither TensorFlow nor scikit-learn found!")
                print("  Install one of them:")
                print("    pip install tensorflow     (best accuracy, ~350 MB)")
                print("    pip install scikit-learn   (lighter, ~30 MB)")
                return

    # Step 1: Download
    print(f"\n[1/3] Downloading EMNIST dataset...")
    download_emnist()

    # Step 2: Load
    print(f"\n[2/3] Loading dataset...")
    train_images = load_emnist_images(os.path.join(DATA_DIR, EMNIST_FILES["train_images"]))
    train_labels = load_emnist_labels(os.path.join(DATA_DIR, EMNIST_FILES["train_labels"]))
    test_images = load_emnist_images(os.path.join(DATA_DIR, EMNIST_FILES["test_images"]))
    test_labels = load_emnist_labels(os.path.join(DATA_DIR, EMNIST_FILES["test_labels"]))
    print(f"  Train: {len(train_images)} | Test: {len(test_images)} | Classes: {NUM_CLASSES}")

    # Step 3: Train
    print(f"\n[3/3] Training model...")
    save_class_mapping()

    if use_tf:
        accuracy = train_tensorflow(train_images, train_labels, test_images, test_labels)
    else:
        accuracy = train_sklearn(train_images, train_labels, test_images, test_labels)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Training complete! Accuracy: {accuracy:.2%}")
    print(f"  Run: python air_keyboard.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
