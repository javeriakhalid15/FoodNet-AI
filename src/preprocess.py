import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from food_info import FOLDER_TO_CLASS, CLASS_NAMES

IMG_SIZE = (100, 100)
BATCH_SIZE = 32


def get_dataset_folders(dataset_root):
    train_dir = os.path.join(dataset_root, "Training")
    test_dir = os.path.join(dataset_root, "Test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training folder not found at: {train_dir}")

    valid_folders = {}
    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)
        if os.path.isdir(folder_path):
            if folder_name in FOLDER_TO_CLASS:
                class_label = FOLDER_TO_CLASS[folder_name]
                if class_label not in valid_folders:
                    valid_folders[class_label] = []
                valid_folders[class_label].append(folder_path)
            else:
                for class_name in CLASS_NAMES:
                    if class_name.lower() in folder_name.lower():
                        if class_name not in valid_folders:
                            valid_folders[class_name] = []
                        valid_folders[class_name].append(folder_path)
                        break

    print(f"Found {len(valid_folders)} matching classes:")
    for cls in valid_folders:
        print(f"  {cls}: {len(valid_folders[cls])} folder(s)")

    return valid_folders, test_dir


def load_images_from_folder(folder_path, label, img_size=IMG_SIZE, max_per_folder=300):
    images = []
    labels = []
    count = 0

    for filename in os.listdir(folder_path):
        if count >= max_per_folder:
            break
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                labels.append(label)
                count += 1
            except Exception as e:
                print(f"  Warning: could not read {img_path}: {e}")

    return images, labels


def load_dataset(dataset_root, img_size=IMG_SIZE, max_per_class=500):
    valid_folders, _ = get_dataset_folders(dataset_root)

    all_images = []
    all_labels = []

    for class_name, folder_list in valid_folders.items():
        class_images = []
        class_labels = []
        for folder_path in folder_list:
            imgs, lbls = load_images_from_folder(
                folder_path, class_name, img_size,
                max_per_folder=max_per_class // len(folder_list)
            )
            class_images.extend(imgs)
            class_labels.extend(lbls)

        print(f"  Loaded {len(class_images)} images for '{class_name}'")
        all_images.extend(class_images)
        all_labels.extend(class_labels)

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels)
    print(f"Total images loaded: {len(X)}")
    return X, y


def preprocess(X, y, save_encoder=True, encoder_path="models/label_encoder.pkl"):
    X = X / 255.0

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y_encoded = le.transform(y)

    from tensorflow.keras.utils import to_categorical
    num_classes = len(CLASS_NAMES)
    y_onehot = to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=0.30, random_state=42, stratify=y_encoded
    )
    y_encoded_temp = np.argmax(y_temp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_encoded_temp
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    if save_encoder:
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, "wb") as f:
            pickle.dump(le, f)
        print(f"Label encoder saved to: {encoder_path}")

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
