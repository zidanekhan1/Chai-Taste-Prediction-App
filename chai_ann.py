import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1️⃣ Load dataset
# ---------------------------
df = pd.read_csv("chai_data.csv")

# ---------------------------
# 2️⃣ Preprocess categorical columns
# ---------------------------
encoders = {}
for col in ["masala_type", "base_type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------
# 3️⃣ Create target column (binary taste)
# ---------------------------
df["taste_label"] = df["taste_level"].apply(lambda x: 1 if x >= 4 else 0)

# ---------------------------
# 4️⃣ Separate features and labels
# ---------------------------
X = df.drop(["taste_level", "taste_label"], axis=1)
y = df["taste_label"]

# ---------------------------
# 5️⃣ Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 6️⃣ Feature scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 7️⃣ Define ANN model
# ---------------------------
model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# 8️⃣ Train model with early stopping
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------
# 9️⃣ Evaluate model
# ---------------------------
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# ---------------------------
# 🔟 Save model and preprocessing tools
# ---------------------------
model.save("chai_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Files saved successfully:")
print(" - chai_model.h5")
print(" - scaler.pkl")
print(" - encoders.pkl")