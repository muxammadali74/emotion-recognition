import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO


face_detection_model = YOLO("models/best.pt")
resnet50_model = resnet50(weights=None)
num_ftrs = resnet50_model.fc.in_features
resnet50_model.fc = nn.Linear(num_ftrs, 7)

resnet50_model.load_state_dict(torch.load("models/model_weights.pth", map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50_model.to(device)
resnet50_model.eval()

cap = cv2.VideoCapture(0)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_detection_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())



            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            try:
                face_tensor = transform(face).unsqueeze(0).to(device)

                outputs = resnet50_model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = classes[predicted.item()]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            except Exception as e:
                print("Ошибка при распознавании эмоции:", e)

    cv2.imshow("Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




