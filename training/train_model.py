import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load features
X = np.load("features/X.npy")
y = np.load("features/y.npy")

with open("features/label_map.json") as f:
    label_map = json.load(f)

num_classes = len(label_map)

# One-hot encode labels
# Train-validation split (use raw labels for stratification)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode AFTER split
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Build Bi-GRU model
model = Sequential([
    Bidirectional(GRU(64, return_sequences=True), input_shape=(200, 40)),
    Dropout(0.3),
    Bidirectional(GRU(32)),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Save model
model.save("models/cough_classifier.h5")

print("✅ Model training completed and saved")
