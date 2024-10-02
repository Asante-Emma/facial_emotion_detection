import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the trained emotion detection model
emotion_model = load_model('../models/facial_emotion_model.keras')

# Initialize MTCNN face detector
face_detector = MTCNN()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face_img, model):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face.astype('float32') / 255.0
    gray_face = np.expand_dims(gray_face, axis=(0, -1))  # Add batch and channel dimensions (1 for grayscale)
    prediction = model.predict(gray_face, batch_size=1)
    emotion = np.argmax(prediction)
    return emotion

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb_frame)
    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)
        face_img = frame[y:y+height, x:x+width]
        if face_img.size > 0:
            emotion = detect_emotion(face_img, emotion_model)
            emotion_label = emotion_labels[emotion]
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Start webcam for real-time emotion detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame
    frame = process_frame(frame)
    
    # Display the frame
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()