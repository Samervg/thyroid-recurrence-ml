import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Veriyi yükleyelim
df = pd.read_csv("thyroid_data.csv")

# Hedef değişkeni belirledik.
y = df["Recurred"].apply(lambda x: 1 if x == "Yes" else 0)

# Bağımsız değişkenler
X = df.drop("Recurred", axis=1)
X = pd.get_dummies(X, drop_first=True)

# Özellik isimlerini kaydedelim
joblib.dump(X.columns.tolist(), "scaler/feature_names.pkl")

# Ölçekleyelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Eğitim/Test ayıralım
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE uygulama kısmı
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# class_weight hesaplama kısmı
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model tanımı
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi
history = model.fit(
    x_train_resampled, y_train_resampled,
    epochs=50,
    batch_size=16,
    validation_data=(x_test, y_test),
    class_weight=class_weight_dict,
    verbose=1
)

# Tahmin ve değerlendirme kısmı
y_pred = (model.predict(x_test) >= 0.5).astype("int32")
print("\n📋 DNN (SMOTE + class_weight) Raporu:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Modeli kaydededelim
model.save("dnn_model_smote.h5")
print("Model başarıyla kaydedildi: dnn_model_smote.h5")
recur_label_map = {"No": 0, "Yes": 1}
joblib.dump(recur_label_map, "label_mapping.pkl")