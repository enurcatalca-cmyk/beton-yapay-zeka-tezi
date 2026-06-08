import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

print("🚀 Dekan sunumu için 1000 satırlık akademik veri seti simüle ediliyor...")
np.random.seed(42)
veri_sayisi = 1000

# Model 1: Dayanım Parametreleri
cimento = np.random.randint(250, 500, veri_sayisi)
katki_kati = np.random.randint(0, 120, veri_sayisi)
su = np.random.randint(140, 220, veri_sayisi)
agrega = np.random.randint(1100, 1800, veri_sayisi)
su_cimento = su / (cimento + katki_kati)
dayanim = (35 / su_cimento) * 0.65 + (katki_kati * 0.08) + np.random.normal(0, 3, veri_sayisi)

# Model 2: İkinci Sayfa İçin Ağır Mühendislik Senaryoları 
# (0: Standart, 1: Kütle Betonu/Hidratasyon Isısı, 2: Sülfat Atağı/Zararlı Çevre, 3: Yüksek Durabilite/Köprü)
senaryo = np.random.randint(0, 4, veri_sayisi)
recete_cimento = np.where(senaryo == 1, cimento - 50, np.where(senaryo == 2, cimento + 30, cimento))
recete_su = np.where(senaryo == 3, su - 20, su)
recete_puzolan = np.where(senaryo == 1, katki_kati + 60, np.where(senaryo == 3, katki_kati + 20, katki_kati))

df_model1 = pd.DataFrame({'cimento': cimento, 'katki': katki_kati, 'su': su, 'agrega': agrega, 'su_cimento_orani': su_cimento, 'dayanim': dayanim})
df_model2 = pd.DataFrame({'senaryo_id': senaryo, 'recete_cimento': recete_cimento, 'recete_su': recete_su, 'recete_puzolan': recete_puzolan})

# Modelleri Disk üzerine Eğiterek Yazma
X1 = df_model1[['cimento', 'katki', 'su', 'agrega', 'su_cimento_orani']]
y1 = df_model1['dayanim']
model1 = XGBRegressor(n_estimators=150, max_depth=6, random_state=42)
model1.fit(X1, y1)
joblib.dump(model1, "beton_model.joblib")

X2 = df_model2[['senaryo_id']]
y2 = df_model2[['recete_cimento', 'recete_su', 'recete_puzolan']]
model2 = XGBRegressor(n_estimators=150, max_depth=6, random_state=42)
model2.fit(X2, y2)
joblib.dump(model2, "recete_model.joblib")

print("🎉 ŞANINIZA YAKIŞIR İKİ YAPAY ZEKA MODELİ DE BAŞARIYLA EĞİTİLDİ!")