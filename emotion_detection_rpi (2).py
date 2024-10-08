import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import os
import time

#Update the model path to point to the directory containing the ensemble models
model_path = ''

# Load the ensemble models
num_models = 5
models = []

for i in range(num_models):
    model_name = f'emotion_detection_model_{i+1}.tflite'
    model_file = model_path + model_name
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    models.append(interpreter)

#Get input and output details for the models
input_details = models[0].get_input_details()
output_details = models[0].get_output_details()

#Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#Load face cascade classifier
face_cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'  # Update this path if necessary
face_cascade = cv2.CascadeClassifier(face_cascade_path)

#Initialize Picamera2 and configure it
picam2 = Picamera2()
picam2.start_preview()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)

#Create a directory to save preprocessed face images
os.makedirs("inference_faces", exist_ok=True)
face_count = 0

#Video settings
video_duration = 30  # seconds
frame_width = 640
frame_height = 480
fps = 20

#Initialize video writer
video_writer = cv2.VideoWriter('emotion_detection_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

try:
    picam2.start()

    def picamera2_to_cv2(picam2_image):
        return cv2.cvtColor(picam2_image, cv2.COLOR_BGRA2BGR)

    start_time = time.time()
    frame_count = 0

    while True:

        frame = picamera2_to_cv2(picam2.capture_array())
        frame = cv2.flip(frame, -1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0

            reshaped_face = np.expand_dims(normalized_face, axis=0)
            reshaped_face = np.expand_dims(reshaped_face, axis=-1).astype(np.float32)

            cv2.imwrite(f"inference_faces/face_{face_count}.png", resized_face * 255)
            face_count += 1
            
            predictions = []
            
            for model in models:
                model.set_tensor(input_details[0]['index'], reshaped_face)
                model.invoke()
                prediction = model.get_tensor(output_details[0]['index'])[0]
                predictions.append(prediction)

            ensemble_prediction = np.mean(predictions, axis=0)
            emotion_idx = np.argmax(ensemble_prediction)
            emotion = emotion_labels[emotion_idx]

            print("Detected emotion:", emotion)  # Debugging line to check detected emotion

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        video_writer.write(frame)
        cv2.imshow('Real-time Emotion Detection', frame)
        frame_count += 1
        
        if frame_count == 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps}")
            frame_count = 0
            start_time = time.time()

        if time.time() - start_time > video_duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    video_writer.release()
