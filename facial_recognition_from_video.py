import face_recognition
import os
import cv2
import time

######
import firebase_admin
from firebase_admin import credentials, firestore
######

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog' daha hızlıdır ama 'cnn' daha doğru sonuç verir (GPU destekliyse)

#####
# Servis hesabı JSON dosyasını yükle
cred = credentials.Certificate("meb-robot-2ded0-firebase-adminsdk-fbsvc-472028b25b.json")
firebase_admin.initialize_app(cred)

# Firestore istemcisi
db = firestore.client()
#####

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Kamera açılamadı.")
    exit()

# İsimden renk üret
def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

print('Loading known faces...')
known_faces = []
known_names = []

# Bilinen yüzleri klasörden yükle
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Uyarı: Yüz bulunamadı -> {image_path}")
            continue
        known_faces.append(encodings[0])
        known_names.append(name)


print('Processing unknown faces...')
while True:
    ret, image = video.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f'Found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        if face_encoding is None:
            continue
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = name_to_color(match)

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            
            docs = db.collection("bina1").get()

            for doc in docs:
                print(f"Belge ID: {doc.id}")
                print("Veri:", doc.to_dict())
                print("---------------")
                if doc.id == match:
                    db.collection("bina1").document(match).delete()
                    print("Çıkış yaptınız.")
                else:
                    db.collection("bina1").document(match).set({
                        "saat":"10.30"
                    })
                    print("Binaya hoşgeldiniz.")
            time.sleep(3)

    # Görüntüyü ekranda göster
    cv2.imshow("Face Recognition", image)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
video.release()
cv2.destroyAllWindows()
