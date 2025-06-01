from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    cap = cv2.VideoCapture(0) 
    detector = HandDetector(maxHands=1)  
    classifier = Classifier(
        r"C:\Users\manya\OneDrive\Desktop\PROJECT\keras_model.h5", 
        r"C:\Users\manya\OneDrive\Desktop\PROJECT\labels.txt"  
    )

    offset = 20  
    imgSize = 300  
    labels = ["Bad", "Good", "GoodBye", "Had Breakfast", "Had Dinner", "Had Lunch",
              "Hello", "Help", "How are you", "Iloveyou", "Lets Play", "No", "Peace",
              "Please", "Thankyou", "Very Good", "Yes"]

    
    while True:
        success, img = cap.read()
        if not success:
            break  
        imgOutput = img.copy()  
        hands, img = detector.findHands(img)  
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset] 

           
            aspectRatio = h / w
            if aspectRatio > 1:  
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:  
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

           
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+10), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 4)

       
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
