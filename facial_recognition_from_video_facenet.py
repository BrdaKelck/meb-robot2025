import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model yükle
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
model_path = 'facenet.pth'
model = InceptionResnetV1(classify=True, num_classes=8631)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Bilinen yüzleri yükle
KNOWN_FACES_DIR = 'known_faces'
known_embeddings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
        img_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)

        if face is not None:
            face = face.unsqueeze(0).to(device)  # 4D tensör (1, 3, 160, 160)
            embedding = model(face).detach().cpu()
            known_embeddings.append(embedding)
            known_names.append(name)

# Kamera başlat
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn(img_rgb)

    if faces is not None:
        if isinstance(faces, torch.Tensor):
            faces = [faces]  # Tek yüz varsa listeye çevir

        for face in faces:
            if face is None:
                continue

            face = face.unsqueeze(0).to(device)
            embedding = model(face).detach().cpu()

            # Karşılaştırma
            min_dist = float('inf')
            matched_name = 'Bilinmiyor'

            for ref_emb, name in zip(known_embeddings, known_names):
                dist = (embedding - ref_emb).norm().item()
                if dist < min_dist and dist < 0.9:
                    min_dist = dist
                    matched_name = name

            # Görüntüye ad yaz
            cv2.putText(frame, matched_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition (Facenet)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
