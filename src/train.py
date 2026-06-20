import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from food_info import CLASS_NAMES


# ===============================
# PATHS (FIXED)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "foodnet_best.keras")
PLOT_DIR = MODEL_DIR

IMG_SHAPE = (100, 100, 3)


# ===============================
# MODEL BUILDING
# ===============================
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Model built: {model.count_params():,} parameters")
    return model, base_model


# ===============================
# FINE TUNING
# ===============================
def unfreeze_top_layers(model, base_model, num_layers=30):
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Fine-tuning enabled: last {num_layers} layers unfrozen")
    return model


# ===============================
# CALLBACKS
# ===============================
def get_callbacks():
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    return [checkpoint, early_stop, reduce_lr]


# ===============================
# TRAINING PLOTS
# ===============================
def plot_history(history1, history2=None):
    acc = history1.history["accuracy"]
    val_acc = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    if history2:
        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150)
    plt.show()


# ===============================
# CONFUSION MATRIX
# ===============================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()


# ===============================
# EVALUATION
# ===============================
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_true, y_pred)

    return acc


# ===============================
# IMPORTANT NOTE
# ===============================
"""
SAVE MODEL CORRECTLY:

Use ONLY this:

model.save("models/foodnet_best.keras")

DO NOT use .h5 anymore
"""