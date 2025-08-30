import cv2
import os

Face_Finder = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

student_name = input("Enter Student Name: ")
save_path = os.path.join("dataset", student_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0
max_images = 150

while True:
    success, frame = video.read()
    if not success:
        print("Frame Nil")
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Face_Finder.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        face_crop = frame[y:y+h, x:x+w]

        count += 1
        file_name = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(file_name, face_crop)

        cv2.putText(frame, f"Images: {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Cam", frame)

    key = cv2.waitKey(1) & 0xFF

    if count >= max_images or key == 13 or key == 27:
        break

video.release()
cv2.destroyAllWindows()

print(f"✅ {count} images saved in {save_path}")
