import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Hyperparameters & Settings
batch_size = 32
img_size = (224, 224)
channels = 3
epochs = 20

# 2. Data Generators
# Note: Ensure train_df, valid_df, test_df exist in your environment
# e.g., train_df = pd.read_csv("train.csv")
tr_gen = ImageDataGenerator(rescale=1.0 / 255)
ts_gen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = tr_gen.flow_from_dataframe(
    train_df,
    x_col="image",
    y_col="label",
    target_size=img_size,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)

valid_gen = ts_gen.flow_from_dataframe(
    valid_df,
    x_col="image",
    y_col="label",
    target_size=img_size,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)

test_gen = ts_gen.flow_from_dataframe(
    test_df,
    x_col="image",
    y_col="label",
    target_size=img_size,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size,
)

classes = list(train_gen.class_indices.keys())
n_classes = len(classes)
print("Class Indices:", train_gen.class_indices)

# 3. Model Architecture
model = Sequential([
    InputLayer(input_shape=(224, 224, 3)),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),
    Dropout(0.3),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    Dense(n_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 4. Model Training
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    verbose=1,
)

# 5. Model Evaluation
ts_length = len(test_df)
test_batch_size = max(
    [ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]
)
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print("-" * 30)
print(f"Train Loss: {train_score[0]:.4f} | Train Acc: {train_score[1]:.4f}")
print(f"Valid Loss: {valid_score[0]:.4f} | Valid Acc: {valid_score[1]:.4f}")
print(f"Test Loss:  {test_score[0]:.4f} | Test Acc:  {test_score[1]:.4f}")
print("-" * 30)

# Predictions & Classification Report
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")
print(classification_report(test_gen.classes, y_pred, target_names=classes))

# 6. ROC Curves
y_true = label_binarize(test_gen.classes, classes=range(n_classes))
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10, 8))
colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue", "navy", "deeppink", "seagreen"])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC: {classes[i]} (AUC = {roc_auc[i]:0.2f})")

plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.2f})', color="deeppink", linestyle=":", linewidth=4)
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

# 7. Save Model
model.save("model2.h5")
print("Saved trained weights to model2.h5")