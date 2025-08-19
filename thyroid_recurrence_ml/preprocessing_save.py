import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

#  Veri okuma kısmı
df = pd.read_csv("thyroid_data.csv")

#  One-hot encoding (SHAP ile uyumlu hale getirme)
X = pd.get_dummies(df.drop("Recurred", axis=1), drop_first=True)
y = df["Recurred"]

#  Ölçekleyelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Kaydetme kısmı
os.makedirs("scaler", exist_ok=True)
joblib.dump(scaler, "scaler/scaler.pkl")
joblib.dump(X.columns.tolist(), "scaler/feature_names.pkl")

print(f"✅ scaler.pkl ve feature_names.pkl başarıyla kaydedildi. Özellik sayısı: {X.shape[1]}")
