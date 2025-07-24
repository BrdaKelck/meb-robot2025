import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    print("Kamera acilamadi.")
    exit()

def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Uyari: Yuz bulunamadi -> {image_path}")
            continue
        known_faces.append(encodings[0])
        known_names.append(name)

print('Processing unknown faces...')
while True:
    ret, frame = video.read()
    if not ret:
        print("Kamera goruntusu alinamadi.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        if face_encoding is None:
            continue
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'- {match} eslesti')

            top, right, bottom, left = [v * 4 for v in face_location]
            color = name_to_color(match)

            cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 22), color, cv2.FILLED)
            cv2.putText(frame, match, (left + 10, bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
