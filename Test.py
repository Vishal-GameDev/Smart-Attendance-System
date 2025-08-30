from fer import FER
import cv2

# Load image
img = cv2.imread("SwathiPic.jpg")

# Initialize emotion detector
detector = FER(mtcnn=True)

# Detect emotions
result = detector.detect_emotions(img)

print(result)  # Gives bounding box + emotions with probabilities
