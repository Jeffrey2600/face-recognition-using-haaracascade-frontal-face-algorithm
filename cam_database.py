import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset='dataset'
name='jeff'

path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(130,100)

# Capture video from webcam
cap = cv2.VideoCapture(0)

count=1

while count<30:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faceOnly=gray[y:y+h,x:x+w]
        resizeImg=cv2.resize(faceOnly,(width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count),resizeImg)
        count+=1

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Image captured Successfully")
# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
