import torch
import cv2
import json

from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from collections import Counter
from datetime import timedelta
from django.http import StreamingHttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView
from django.utils import timezone
from .models import EmployeeLog

camera_running = True

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SigLIP2 model and processor
model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name).to(device)
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

labels = {
    "0": "Ahegao", "1": "Angry", "2": "Happy", "3": "Neutral",
    "4": "Sad", "5": "Surprise"
}

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_emotion(face_img):
    """Predict emotion from a face image using SigLIP2."""
    image = Image.fromarray(face_img).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Get predicted emotion label
    pred_index = probs.index(max(probs))
    return labels[str(pred_index)]


def gen_frames():
    """Generator function for streaming video frames with emotion recognition."""
    global camera_running
    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            status = "Present"
            color = (0, 255, 0)
        else:
            status = "Not Present"
            color = (0, 0, 255)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion_label = predict_emotion(face_img)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{emotion_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display overall presence status
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Store log in DB
        EmployeeLog.objects.create(
            timestamp=timezone.now(),
            status=status,
            emotion=emotion_label if len(faces) > 0 else "N/A",
        )

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


class HomeView(TemplateView):
    template_name = "app/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        today = timezone.localtime(timezone.now()).date()
        week_ago = today - timedelta(days=7)

        # Filter logs for today and week
        logs_today = EmployeeLog.objects.filter(timestamp__date=today).order_by('timestamp')
        logs_week = EmployeeLog.objects.filter(timestamp__date__gte=week_ago).order_by('timestamp')

        # Presence timeline
        timeline = [{"time": log.timestamp.strftime("%H:%M"), "status": log.status} for log in logs_today]

        # Emotion distribution
        emotions_today = [log.emotion for log in logs_today]
        emotion_counts = Counter(emotions_today)
        emotion_labels = list(emotion_counts.keys())
        emotion_values = list(emotion_counts.values())

        # Daily summary
        total_logs = len(logs_today)
        present_logs = sum(1 for log in logs_today if log.status == "Present")
        presence_percent = round((present_logs / total_logs * 100), 2) if total_logs > 0 else 0
        most_common_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "N/A"

        # Weekly summary
        weekly_emotions = [log.emotion for log in logs_week]
        weekly_emotion_counts = Counter(weekly_emotions)
        most_common_weekly_emotion = weekly_emotion_counts.most_common(1)[0][0] if weekly_emotion_counts else "N/A"

        context.update({
            "timeline": timeline,
            "emotion_labels": json.dumps(emotion_labels),
            "emotion_values": json.dumps(emotion_values),
            "presence_percent": presence_percent,
            "most_common_emotion": most_common_emotion,
            "most_common_weekly_emotion": most_common_weekly_emotion,
        })
        return context


class CameraView(TemplateView):
    template_name = "app/camera.html"


def video_feed(request):
    global camera_running
    camera_running = True
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def stop_camera(request):
    global camera_running
    camera_running = False
    return redirect("home")
