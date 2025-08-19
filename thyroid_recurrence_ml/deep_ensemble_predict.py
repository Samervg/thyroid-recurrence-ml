import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 🔹 Modelleri ve yardımcı dosyaları yükle
rf_model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("scaler/scaler.pkl")
feature_names = joblib.load("scaler/feature_names.pkl")
dnn_model = load_model("dnn_model_smote.h5")


# 🔸 Kullanıcıdan manuel veri girişi
input_data = {
    "Age": 55,
    "Gender": "M",
    "Smoking": "Yes",
    "Hx Smoking": "No",
    "Hx Radiothreapy": "Yes",
    "Focality": "Multifocal",
    "Laterality": "Bilateral",
    "Tumor Size": 3.4,
    "Tumor Stage": "T3",
    "Lymph Node Stage": "N1b",
    "Metastasis Stage": "M0",
    "TNM Stage": "Stage III",
    "ATA Risk": "High",
    "Response": "Structural Incomplete",
    "Adenopathy": "Right",
    "Histology": "Papillary"
}

# 🔹 DataFrame'e çevir
df_input = pd.DataFrame([input_data])

# 🔹 Kategorik verileri get_dummies ile encode et
df_encoded = pd.get_dummies(df_input, drop_first=True)

# 🔹 Eksik sütunları ekle (modelde olup kullanıcı verisinde olmayanlar)
for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# 🔹 Sıra garanti altına alınır
df_encoded = df_encoded[feature_names]

# 🔹 Ölçekleme
X_scaled = scaler.transform(df_encoded)

# 🔹 Random Forest tahmini (olasılık)
rf_proba = rf_model.predict_proba(df_encoded)[0][1]  # sınıf 1 (nüks)

# 🔹 DNN tahmini (olasılık)
dnn_proba = dnn_model.predict(X_scaled)[0][0]  # tek değer

# 🔹 Ensemble ortalaması
ensemble_score = (rf_proba + dnn_proba) / 2

# 🔹 Sonuç yorumu
print(f"\n🔍 RF tahmini: {rf_proba:.2f}")
print(f"🔍 DNN tahmini: {dnn_proba:.2f}")
print(f"🧠 Ensemble ortalama skor: {ensemble_score:.2f}")

if ensemble_score >= 0.5:
    print("📢 Tahmin: Bu hasta NÜKS EDEBİLİR.")
else:
    print("📢 Tahmin: Bu hasta NÜKS ETMEYEBİLİR.")
