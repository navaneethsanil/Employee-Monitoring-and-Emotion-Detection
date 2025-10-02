from django.urls import path
from .views import HomeView, CameraView, video_feed, stop_camera

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("camera/", CameraView.as_view(), name="camera"),
    path("video_feed/", video_feed, name="video_feed"),
    path("stop_camera/", stop_camera, name="stop_camera"),
]
