import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 🔹 Veriyi yükle
df = pd.read_csv("thyroid_data.csv")
X = pd.get_dummies(df.drop("Recurred", axis=1), drop_first=True)
y = df["Recurred"]

# 🔹 Özellik uyumu kontrolü
feature_names = joblib.load("scaler/feature_names.pkl")
if list(X.columns) != feature_names:
    raise ValueError("Sütun isimleri feature_names.pkl ile eşleşmiyor.")
# Hedef değişkeni 0 ve 1 olacak şekilde dönüştür
y = y.replace({"No": 0, "Yes": 1}).astype(np.float32)

# 🔹 SMOTE uygulama
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
y_resampled = y_resampled.astype(np.float32)

# 🔹 Ölçekleme
scaler = joblib.load("scaler/scaler.pkl")
X_scaled = scaler.transform(X_resampled)

# 🔹 Train-test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 🔸 Basit Attention mekanizması
def attention_block(inputs):
    attention_weights = Dense(inputs.shape[1], activation='softmax')(inputs)
    attention_output = Multiply()([inputs, attention_weights])
    return attention_output

# 🔹 Model mimarisi
input_dim = X_train.shape[1]
inputs = Input(shape=(input_dim,))
x = Dense(128, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# 🔸 Attention katmanı
x = attention_block(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 🔹 Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 🔹 Klasör yoksa oluştur
import os
os.makedirs("models", exist_ok=True)

# 🔹 Modeli kaydet
model.save("models/dnn_model_attention.h5")
print("✅ Attention'lı DNN modeli başarıyla kaydedildi: models/dnn_model_attention.h5")
