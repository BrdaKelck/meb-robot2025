import firebase_admin
from firebase_admin import credentials, firestore

# Servis hesabı JSON dosyasını yükle
cred = credentials.Certificate("meb-robot-2ded0-firebase-adminsdk-fbsvc-472028b25b.json")
firebase_admin.initialize_app(cred)

# Firestore istemcisi
db = firestore.client()

# Örnek veri
data = {
    'saat': '10.10',
    
}

# 'ogrenciler' koleksiyonuna veri ekle
db.collection('bina1').document("berat").set(data)

print("Veri Firestore'a eklendi.")

# "kullanicilar" koleksiyonundaki tüm belgeleri çek
docs = db.collection("bina1").get()

for doc in docs:
    print(f"Belge ID: {doc.id}")
    print("Veri:", doc.to_dict())
    print("---------------")

