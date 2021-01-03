import cv2
import time
import numpy as np
import tensorflow.keras
from pygame import mixer
from PIL import Image, ImageOps

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(1)

global sayi
global model
global sonuc
sonuc = False

mixer.init()

np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

def classify():
    global sonuc
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open('run.jpg')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    label = 0
    while label < 2:
        if prediction[:,[label]] > 0.6:
            if label == 0:
                mixer.music.load('abep.mp3')
                mixer.music.play()
                time.sleep(0.1)
                sonuc = True
            elif label == 1:
                sonuc = False
        label = label + 1

    if sonuc == True:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cikti = frame[y:y+h, x:x+w]
        cv2.imwrite('run.jpg', cikti)
        classify()
        print(sonuc)

        cv2.imshow('ben', cikti)

    cv2.imshow('wow', frame)

    #ESC Pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
