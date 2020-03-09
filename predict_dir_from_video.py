import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

labels = ['right', 'left', 'up', 'down']

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

video_capture = cv2.VideoCapture(1)

process_this_frame = True

while True:
    
    ret, frame = video_capture.read()
    size = (224, 224)
    small_frame = cv2.resize(frame, size)

    if process_this_frame:
        
        image_array = np.asarray(small_frame)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)

    process_this_frame = not process_this_frame

    font = cv2.FONT_HERSHEY_DUPLEX
    txt = labels[prediction.argmax()]
    cv2.putText(frame, txt, (116,90), font, 2.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
