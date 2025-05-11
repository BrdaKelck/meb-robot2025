# 🏠 Disaster Response Face Recognition System

## 📌 Project Purpose

The primary objective of this project is to quickly inform authorized emergency teams about the **number** and **identity** of individuals inside a collapsed building during a potential disaster.

## ⚙️ How It Works

- **Cameras** are installed at the entrances of each building and perform **real-time facial recognition**.
- Each building has a set of **pre-registered faces** corresponding to its residents.
- When a face is detected:
  - ✅ If it **matches a registered resident**, the system logs their **identity** and adds them to the **Current Occupants** list.
  - 🆕 If it **does not match** any known resident, the system treats it as a **guest**, stores the face in the **Unknown Faces** folder, and still adds them to the Current Occupants list.
- When a person is detected again on exit, they are **removed** from the Current Occupants list.
- This enables continuous, real-time tracking of who is inside the building at any given moment.

## 📁 Folder Structure

project_root/
├── known_faces/
│ ├── host1/
│ │ ├── face1.jpg
│ │ ├── face2.jpg
│ │ └── ...
│ ├── host2/
│ │ └── ...
│ └── ...
│
├── unknown_faces/
│ ├── guest1/
│ │ ├── face1.jpg
│ │ ├── face2.jpg
│ │ └── ...
│ ├── guest2/
│ │ └── ...
│ └── ...


🔍 Firestore Database Structure
We are using Cloud Firestore to manage and track the presence of recognized faces.

📁 Collection: bina1
This is the main collection representing a physical building or area, such as a school entrance, office gate, etc.

📄 Documents (Inside bina1 collection)
Each document represents an individual person (recognized by face recognition) entering or exiting the building. The document ID corresponds to the person's name or generated guest ID (e.g. "john_doe", "guest0", "guest1"...).

Example:

bina1 (collection)
├── guest0 (document)
│   └── saat: "10.30"
├── alice (document)
│   └── saat: "10.45"
├── guest1 (document)
│   └── saat: "11.00"


📌 Fields in Each Document:
saat: A string value representing the entry time (e.g., "10.30"). You can later extend this to include more fields like entry_time, exit_time, or camera_id as needed.

This structure allows us to:

-Check if a person is already in the building by checking if their document exists in the bina1 collection.

-Add a new entry when someone enters.

-Delete the document when the same person is recognized again (indicating they are leaving).

## 💻 System Requirements

- Python 3.8+
- OpenCV
- face_recognition
- dlib
- NumPy
- Compatible camera (USB or IP cam)

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run the main application
python main.py


🚨 Use Case
This system can be deployed in residential buildings, dormitories, or other public housing facilities to support emergency response units during natural disasters such as earthquakes.

📬 Contact
For questions or suggestions, please contact:
📧 beratkklck@gmail.com
