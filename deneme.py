import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

# Ayarlar
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.7  # cosine benzerliği için eşik
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
EMBEDDING_SIZE = 512

# Cihaz seçimi (GPU varsa cuda, yoksa cpu kullanılır)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modeller
mtcnn = MTCNN(keep_all=False, device=device)
model_path = 'facenet.pth'  # Modelin yolunu belirtin
model = InceptionResnetV1(classify=True, num_classes=8631)  # Eğer num_classes belirliyorsanız
model.load_state_dict(torch.load(model_path, map_location=device))  # Modeli yükleyin
model.eval()

# Yüz isimlerine göre renk oluşturma fonksiyonu
def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

# Known faces yükleme
print("Yüzler yükleniyor...")
known_embeddings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
        path = os.path.join(KNOWN_FACES_DIR, name, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB formatına çevir
        faces = mtcnn(img_rgb)
        
        if faces is None:
            print(f"Uyari: Yüz bulunamadı -> {path}")
            continue
        
        embedding = model(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()  # Yüz embedding'ini al
        known_embeddings.append(embedding)
        known_names.append(name)

print("Kamera açılıyor...")
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB formatına çevir
    faces, _ = mtcnn.detect(img_rgb)  # Yüz tespiti

    if faces is not None:
        for face in faces:
            left, top, right, bottom = [int(coord) for coord in face]  # Yüzün koordinatlarını alıyoruz

            # Geçerli koordinatlar kontrolü ekleyelim
            if left < 0 or top < 0 or right > frame.shape[1] or bottom > frame.shape[0]:
                continue  # Eğer yüz koordinatları geçerli değilse, o yüzü geç

            # Yüzün çevresine dikdörtgen çiz
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)

            # Yüzün alanını kopyala
            face_cropped = frame[top:bottom, left:right]

            # Yüz bölgesinin boş olup olmadığını kontrol et
            if face_cropped.size == 0:
                continue

            # Yüzü RGB formatına dönüştür
            face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)

            # Yüzü 160x160 boyutlarına indir
            face_rgb_resized = cv2.resize(face_rgb, (160, 160))  # Model 160x160 boyutunda giriş bekler

            # Yüzü tensöre dönüştürüp kanal sırasını uygun hale getirin
            face_tensor = torch.from_numpy(face_rgb_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (3, 160, 160)

            # Yüz embedding'ini al
            embedding = model(face_tensor).detach().cpu().numpy()

            # Tanıma kısmı
            min_dist = float("inf")
            match_name = None

            for idx, known_embedding in enumerate(known_embeddings):
                dist = np.linalg.norm(embedding - known_embedding)
                if dist < min_dist:
                    min_dist = dist
                    match_name = known_names[idx]

            if min_dist < TOLERANCE:
                cv2.putText(frame, match_name, (left + 10, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), FONT_THICKNESS)
            else:
                cv2.putText(frame, "Bilinmiyor", (left + 10, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), FONT_THICKNESS)

    # Kameradan alınan görüntüyü göster
    cv2.imshow("Facenet Yüz Tanıma", frame)

    # 'q' tuşuna basarak çıkabiliriz
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
