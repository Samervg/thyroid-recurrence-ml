import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#  Model ve scaler dosyalarını yükle
model = load_model("models/dnn_model_attention.h5")
scaler = joblib.load("scaler/scaler.pkl")
feature_names = joblib.load("scaler/feature_names.pkl")

#  Veriyi yükleyip ve işleyelim
df = pd.read_csv("thyroid_data.csv")
X = pd.get_dummies(df.drop("Recurred", axis=1), drop_first=True)

#  Sütun kontrolü
if list(X.columns) != feature_names:
    raise ValueError("Veri sütunları, feature_names.pkl ile uyuşmuyor.")

#  Ölçekleyelim
X_scaled = scaler.transform(X)

# 🔹 SHAP DeepExplainer ile açıklama
explainer = shap.DeepExplainer(model, X_scaled[:100])
shap_values = explainer.shap_values(X_scaled[:100])

# SHAP değerini işleyelim
if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

#  Ortalama SHAP değerlerini al ve düz float listesine dönüştür
mean_shap = np.mean(np.abs(shap_vals), axis=0)
mean_shap = [float(val) for val in mean_shap]  # 🔥 En kritik satır: her değeri float yap

# 8. Özellikleri sırala
sorted_idx = np.argsort(mean_shap)[::-1]
sorted_features = [str(f) for f in np.array(feature_names)[sorted_idx]]  # string'e zorla
sorted_shap_vals = [mean_shap[i] for i in sorted_idx]  # sıralı float listesi

#  9. Bar grafik çiz
plt.figure(figsize=(10, 8))
plt.barh(sorted_features[:20][::-1], sorted_shap_vals[:20][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.show()

#   PNG olarak kaydetmek istersek:
# plt.savefig("shap_bar_dnn_attention.png", dpi=300, bbox_inches="tight")
