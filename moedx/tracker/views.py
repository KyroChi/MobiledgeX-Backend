from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from tracker.serializers import DetectedSerializer
from facial_detection.facedetector import FaceDetector

import cv2

# Create your views here.

@csrf_exempt
@fd_decorator
def frame_detect(request):
    """
    Runs facial detection on a frame that is sent via a REST
    API call
    """
    return HttpResponse(fd.test())


def fd_decorator():
    def _meta(*args, **kwargs):
        fd = FaceDetector()
        return frame_detect(*args, **kwargs):
    return _meta
    
