import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")  

df = pd.read_csv("thyroid_data.csv")
df.head()
print(df.head())

print("\n[VERİ TİPLERİ]")
print(df.info())

print("\n[EKSİK VERİLER]")
print(df.isnull().sum())

print("\n[RECURRED DAĞILIMI]")
print(df["Recurred"].value_counts())


sns.countplot(x="Recurred", data=df)
plt.title("Recurred (nüks) Dağilimi") 
plt.xlabel("Nüks")                       # nüks dağilimi
plt.ylabel("Hasta Sayisi")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", kde=True, bins=20)
plt.title("Yaş Dağilimi")                          # yaş dagilimi
plt.xlabel("Yaş")
plt.ylabel("Hasta Sayisi")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Recurred", y="Age", data=df)       
plt.title("Recurred (Nüks) - Yaş İlişkisi")         # nükse göre yaş kutu grafigi
plt.xlabel("Nüks")
plt.ylabel("Yaş")
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(x="Gender", hue="Recurred", data=df)
plt.title("Cinsiyete Göre Nüks dağilimi")            
plt.xlabel("Cinsiyet")                              # cinsiyete göre nüks dağilimi
plt.ylabel("Hasta Sayisi")
plt.legend(title="Nüks")
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(x="Risk", hue="Recurred", data=df)
plt.title("Risk Sinifina Göre Nüks Dağilimi")       # risk sınıfına göre nüks dağilimi
plt.xlabel("ATA Risk Sinifi")
plt.ylabel("Hasta Sayisi")
plt.legend(title="Nüks")
plt.show()

ct = pd.crosstab(df["Pathology"], df["Risk"])
ct.plot(kind="bar", stacked=True, figsize=(8, 5),colormap="Paired")

plt.title("Pathology Türüne Göre Risk Dağilimi (Hasta Sayisi)")    
plt.xlabel("Tümör Tipi (Pathology)")                            # pathology vs risk sayı bazlı
plt.ylabel("Hasta Sayisi")
plt.legend(title="Risk Sinifi")
plt.tight_layout()
plt.show()
 
ct_pct = ct.div(ct.sum(axis=1), axis=0)  # her satırı yüzdeye ceviriyor unutma
ct.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set1")
plt.title("pathology Türüne Göre Risk Dağilimi (Yüzde)")
plt.xlabel("Tümör Tipi (Pathology)")                           
plt.ylabel("Oran")                                         # pathology vs risk yüzde bazlı
plt.legend(title="Risk Sinifi")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Stage", y="Age", data=df)
plt.title("Stage ile Yaş ilişkisi")          #  evre vs yaş
plt.xlabel("Klinik Evre")
plt.ylabel("Yaş")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(x="Response", hue="Recurred", data=df)

plt.title("Tedavi Yanitina Göre Nüks Dağilimi")     
plt.xlabel("Tedaviye Yanit (Response)")
plt.ylabel("Hasta Sayisi")                # response(yanit) vs recurred(nüks)
plt.legend(title="Nüks")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(x="Pathology", hue="Recurred", data=df)
plt.title("Tümör Tipine Göre Nüks Dağilimi")
plt.xlabel("Tümör Tipi (Patoloji)")             # tümör tipi nüks ilişkisi
plt.ylabel("Hasta Sayisi")
plt.legend(title="Nüks")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
