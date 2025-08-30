import face_recognition 
import pickle
import cv2 
import os

dataset_path = "dataset"
print("🔍 Starting to process images in the dataset...")

known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_path):
        continue

    print(f"-> Processing images for: {person_name}")
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        image = cv2.imread(image_path)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        boxes = face_recognition.face_locations(rgb_image, model='hog')
        
        # Compute the facial embedding for the face.
        # This creates the unique 128-dimensional face encoding.
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        
        # Add each found encoding and the corresponding name to our lists.
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the encodings and names to a file for later use.
print("\n💾 Saving encodings to disk...")
data = {"encodings": known_encodings, "names": known_names}

# Open the file in binary write mode and dump the data.
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("✅ Training complete. Encodings saved to 'encodings.pickle'")