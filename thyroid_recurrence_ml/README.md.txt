# Tiroid Kanseri Nüks Tahmini (Thyroid Recurrence Prediction)

Bu proje, tiroid kanseri hastalarında nüks (tekrar etme) riskini tahmin etmek için klasik makine öğrenmesi (ML) ve derin öğrenme (DL) tabanlı yöntemleri karşılaştırmalı olarak incelemektedir. Çalışmada, hem açıklanabilirlik hem de yüksek doğruluk hedeflenmiş, özellikle **Attention mekanizması entegre edilmiş DNN (DNN-Attention)** modeli öne çıkmıştır.  

## 📊 Veri Seti
- 383 hasta verisi
- 17 klinik özellik (yaş, tümör evresi, histoloji, lenf nodu durumu vb.)
- Hedef değişken: `Recurred` (Yes = nüks var, No = nüks yok)

## 🚀 Kullanılan Modeller
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Ensemble yöntemleri (Voting, Stacking)
- Multilayer Perceptron (MLP)
- Derin Sinir Ağı (DNN)
- **DNN + Attention (önerilen model)**

## ⚙️ Kullanılan Teknolojiler
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn (ML algoritmaları)
- tensorflow / keras (DNN ve Attention)
- imbalanced-learn (SMOTE yöntemi)
- SHAP (model açıklanabilirliği)

## 🔎 Veri Ön İşleme
- Eksik veriler incelendi ve temizlendi
- Kategorik değişkenler one-hot encoding ile dönüştürüldü
- Sayısal değişkenler StandardScaler ile ölçeklendirildi
- Sınıf dengesizliği **SMOTE** ile giderildi

## 📈 Sonuçlar
| Model | Doğruluk (Accuracy) | Duyarlılık (Recall) | F1-Skoru |
|-------|----------------------|---------------------|----------|
| Logistic Regression | 96.1% | 95% | 0.95 |
| Decision Tree | 90.9% | 91% | 0.89 |
| Random Forest | 96.1% | 93% | 0.95 |
| Voting Classifier | 97.4% | 95% | 0.97 |
| Stacking Classifier | 96.1% | 95% | 0.95 |
| MLP | 98.0% | 96% | 0.98 |
| DNN | 97.4% | 97% | 0.97 |
| **DNN-Attention** | **98.1%** | **98%** | **0.98** |

📌 **Sonuç:** DNN-Attention modeli, hem en yüksek doğruluk hem de pozitif sınıf (nüks) tespitinde en iyi duyarlılığı sağlamıştır. Ayrıca SHAP analizi ile **Age** ve **Response_Structural Incomplete** değişkenleri en kritik faktörler olarak öne çıkmıştır.

