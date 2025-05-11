import face_recognition
import os
import cv2
import time

# Firebase setup
import firebase_admin
from firebase_admin import credentials, firestore

# Constants
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'  # Directory for guests (unrecognized faces)
TOLERANCE = 0.6  # Lower means stricter matching
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog' is faster, 'cnn' is more accurate but requires GPU

# Load Firebase service account
cred = credentials.Certificate("your_file.json")
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Initialize camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Camera could not be opened.")
    exit()

# Generate a unique color based on the name
def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

# Counter for generating guest IDs
guest_counter = 0

# Load known faces
print('Loading known faces...')
known_faces = []
known_names = []

# Load all faces under each registered person's folder
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Warning: No face found in -> {image_path}")
            continue
        known_faces.append(encodings[0])
        known_names.append(name)

# Load previously recorded unknown (guest) faces
unknown_faces = []
unknown_names = []

for name in os.listdir(UNKNOWN_FACES_DIR):
    for filename in os.listdir(f'{UNKNOWN_FACES_DIR}/{name}'):
        path = os.path.join(UNKNOWN_FACES_DIR, name, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Warning (unknown): No face found in -> {path}")
            continue
        unknown_faces.append(encodings[0])
        unknown_names.append(name)

        # Track highest guest number to avoid duplicate guest IDs
        if name.startswith("guest"):
            try:
                number = int(name[5:])
                guest_counter = max(guest_counter, number + 1)
            except:
                continue

print('Processing video stream...')
while True:
    ret, image = video.read()
    if not ret:
        print("Failed to retrieve frame from camera.")
        break

    # Detect face locations and encodings
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f'Found {len(encodings)} face(s)')

    processed_faces = []  # Avoid reprocessing the same face in a single frame

    for face_encoding, face_location in zip(encodings, locations):
        if face_encoding is None:
            continue

        if face_encoding in processed_faces:
            continue

        # Try to match with known faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - Recognized: {match}')
        else:
            # Try to match with existing unknown (guest) faces
            results_unknown = face_recognition.compare_faces(unknown_faces, face_encoding, TOLERANCE)
            if True in results_unknown:
                match = unknown_names[results_unknown.index(True)]
                print(f' - Recognized guest: {match}')
            else:
                # Completely new face, save to unknown_faces directory
                match = f'guest{guest_counter}'
                guest_dir = os.path.join(UNKNOWN_FACES_DIR, match)
                os.makedirs(guest_dir, exist_ok=True)
                filename = f"{int(time.time())}.jpg"
                cv2.imwrite(os.path.join(guest_dir, filename), image)

                unknown_faces.append(face_encoding)
                unknown_names.append(match)
                guest_counter += 1
                print(f' - New guest added: {match}')

        # Draw a rectangle and name around the detected face
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = name_to_color(match)

        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Check Firestore: if person is already in the building, remove them (exit), otherwise add them (enter)
        docs = db.collection("bina1").get()
        isIn = False
        for doc in docs:
            if doc.id == match:
                isIn = True

        if isIn:
            db.collection("bina1").document(match).delete()
            print("Exit recorded.")
            time.sleep(3)
        else:
            db.collection("bina1").document(match).set({
                    "saat": "10.30"
                })
            print("Entry recorded.")
            time.sleep(3)

        processed_faces.append(face_encoding)

    # Show the current frame
    cv2.imshow("Face Recognition", image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and destroy OpenCV windows
video.release()
cv2.destroyAllWindows()
