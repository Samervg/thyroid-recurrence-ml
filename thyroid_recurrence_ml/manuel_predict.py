import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Model ve scaler yükleme
model = load_model("dnn_model_smote.h5")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("scaler/feature_names.pkl")

# Scaler'dan yaşın normalize edilmesi için ortalama ve std çekiliyor
age_mean = scaler.mean_[feature_names.index("Age")]
age_std = np.sqrt(scaler.var_[feature_names.index("Age")])

# Kullanıcıdan giriş al
print("Lütfen bilgileri giriniz:")

age = float(input("Yaş: ").strip())
gender = input("Cinsiyet (M/F): ").strip().upper()
tumor_type = input("Tümör Tipi (Papillary / Follicular / Micropapillary / Hurthel cell): ").strip().capitalize()
stage = input("Evre (I / II / III / IV): ").strip().upper()
ata_risk = input("ATA Risk (Low / Intermediate / High): ").strip().capitalize()
response = input("Tedavi Yanıtı (Excellent / Indeterminate / Incomplete): ").strip().capitalize()
adenopathy = input("Adenopathy (Right / Left / Posterior / Extensive / No): ").strip().capitalize()
focality = input("Focality (Unifocal / Multifocal): ").strip().capitalize()
hx_radio = input("Radyoterapi Geçmişi (Yes / No): ").strip().capitalize()
hx_smoke = input("Sigara Geçmişi (Yes / No): ").strip().capitalize()

# DataFrame oluştur
manual_data = {
    "Age": 0,  # Geçici olarak sıfır veriyoruz, scale sonrası düzeltilecek
    "Gender": gender,
    "Pathology": tumor_type,
    "Stage": stage,
    "Risk": ata_risk,
    "Response": response,
    "Adenopathy": adenopathy,
    "Focality": focality,
    "Hx Radiothreapy": hx_radio,
    "Hx Smoking": hx_smoke,
    "Smoking": hx_smoke  # Aynı değeri giriyoruz
}

manual_df = pd.DataFrame([manual_data])

# One-hot encoding ve kolon hizalama
manual_df = pd.get_dummies(manual_df, drop_first=True)
manual_df = manual_df.reindex(columns=feature_names, fill_value=0)

# Şimdi yaş değerini kendimiz normalize edip yerleştiriyoruz
manual_df["Age"] = (age - age_mean) / age_std

# Ölçeklenmiş inputu modele veriyoruz
scaled_input = manual_df.values
prediction = model.predict(scaled_input)[0][0]

# Sonucu yazdır
if prediction >= 0.5:
    print(f"\n Tahmin: Bu hasta NÜKS EDEBİLİR. (Skor: {prediction:.2f})")
else:
    print(f"\n Tahmin: Bu hasta NÜKS ETMEYEBİLİR. (Skor: {prediction:.2f})")
recur_map = joblib.load("label_mapping.pkl")
print(recur_map)  # {'No': 0, 'Yes': 1}
