import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

#  Model ve veri
model = joblib.load("models/random_forest.pkl")
feature_names = joblib.load("scaler/feature_names.pkl")

df = pd.read_csv("thyroid_data.csv")
X = pd.get_dummies(df.drop("Recurred", axis=1), drop_first=True)

if list(X.columns) != feature_names:
    raise ValueError("Sütun isimleri uyuşmuyor.")

#  SHAP açıklayıcı
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#  shape kontrol
print("✔ shap_values shape:", shap_values.shape)

#  Bar grafiği (nüks eden sınıf için)
plt.figure(figsize=(12, len(feature_names) * 0.4))
shap.summary_plot(shap_values[:, :, 1], X, plot_type="bar", max_display=len(feature_names), show=True)
