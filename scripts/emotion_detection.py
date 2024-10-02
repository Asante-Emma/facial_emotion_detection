import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import argparse

# Load the trained emotion detection model
emotion_model = load_model('../models/facial_emotion_model.keras')

# Initialize MTCNN face detector
face_detector = MTCNN()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face_img, model):
    """Detect emotion from a face image."""
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face.astype('float32') / 255.0
    gray_face = np.expand_dims(gray_face, axis=(0, -1))  # Add batch and channel dimensions (1 for grayscale)
    prediction = model.predict(gray_face, batch_size=1)
    emotion = np.argmax(prediction)
    return emotion

def process_image(image_path, output_image_path, output_csv_path):
    """Process the image, detect faces, annotate emotions, and save results."""
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert image to RGB (MTCNN requires RGB format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = face_detector.detect_faces(rgb_image)
    
    # Initialize emotion frequency counter
    emotion_counts = {label: 0 for label in emotion_labels}
    
    # If no faces are detected
    if not faces:
        print("No faces detected in the image.")
        return
    
    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)
        face_img = image[y:y+height, x:x+width]
        
        # Check if face image is valid
        if face_img.size > 0:
            # Predict emotion
            emotion = detect_emotion(face_img, emotion_model)
            emotion_label = emotion_labels[emotion]
            
            # Update emotion frequency
            emotion_counts[emotion_label] += 1
            
            # Draw rectangle and label on the image
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save the annotated image
    cv2.imwrite(output_image_path, image)
    
    # Save emotion frequencies to a CSV file
    df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Frequency'])
    df.to_csv(output_csv_path, index=False)
    
    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Emotion Detection Results')
    plt.axis('off')
    plt.show()

'''if __name__ == "__main__":
    # Path to the image file
    image_path1 = '../photos/tonl_founders_pano_349214.jpg'
    image_path2 = '../photos/wednesday-netflix-112922-2-6726157c0fb6404190040756f40da790.jpg'
    image_path3 = '../photos/mature-woman-and-a-young-teenage-girl-mother-and-her-daughter-on-the-banks-of-a-wide-river-MINF14889.jpg'
    image_path4 = '../photos/Wednesday-First-Look-04.webp'
    
    # Output paths
    output_image_path = 'output_image.png'
    output_csv_path = 'emotion_frequencies.csv'
    
    # Process the image
    process_image(image_path1, output_image_path, output_csv_path)'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for facial emotion detection.")
    
    # Define arguments for input image, output image, and output CSV
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_image', type=str, required=True, help="Path to save the output image with annotations.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the emotion frequency CSV file.")
    
    args = parser.parse_args()
    
    # Process the image using the provided arguments
    process_image(args.image, args.output_image, args.output_csv)
