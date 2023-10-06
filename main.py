import os
from ultralytics import YOLO
import cv2
from pushbullet import Pushbullet


# Import the following modules


# Function to send Push Notification

def notif(msg):
    ph_key="o.bYeOTNoJX5lNhZM10ooLMFommYCRqMPa"
    pb=Pushbullet(ph_key)
    phone = pb.devices[0]
    pb.push_sms(phone, "+919489837945",msg)



video_path = 'fire_2.mp4'
video_path_out = 'fire2_out.webm'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _= frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'VP90'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.50
c=0
while ret:
    if(c>2):
        notif("Fire Detected Alert!!")
        print('Message sent!!')
        break
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            c+=1
    out.write(frame)
    ret, frame = cap.read()
cap.release()
out.release()
cv2.destroyAllWindows()
