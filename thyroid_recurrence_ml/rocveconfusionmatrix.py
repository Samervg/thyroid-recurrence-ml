import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. DNN-Attention Modelini Yükle ---
model = load_model("models/dnn_model_attention.h5")

# Scaler ve feature_names yükle
scaler = joblib.load("scaler/scaler.pkl")
feature_names = joblib.load("scaler/feature_names.pkl")

# --- 2. Veriyi Hazırla ---
df = pd.read_csv("thyroid_data.csv")
X = pd.get_dummies(df.drop("Recurred", axis=1), drop_first=True)
y = df["Recurred"].replace({"No": 0, "Yes": 1}).astype(float)

# Feature sırasını koru
X = X[feature_names]
X_scaled = scaler.transform(X)

# Train-test ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Voting Classifier Modelini Eğit ---
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('dt', dt)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# --- 4. ROC Eğrisi ---
# DNN-Attention olasılık tahminleri
y_proba_dnn = model.predict(X_test).ravel()

# Voting Classifier olasılık tahminleri
y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]

# ROC verileri
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_proba_dnn)
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_proba_voting)

roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
roc_auc_voting = auc(fpr_voting, tpr_voting)

# ROC çizim
plt.figure(figsize=(8, 6))
plt.plot(fpr_dnn, tpr_dnn, label=f"DNN-Attention (AUC = {roc_auc_dnn:.2f})")
plt.plot(fpr_voting, tpr_voting, label=f"Voting Classifier (AUC = {roc_auc_voting:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi Karşılaştırması")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 5. Confusion Matrix (DNN-Attention) ---
y_pred_dnn = (y_proba_dnn >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_dnn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - DNN-Attention")
plt.show()
