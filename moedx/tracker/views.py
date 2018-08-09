from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from tracker.serializers import DetectedSerializer
from facial_detection.facedetector import FaceDetector
from django.http import HttpResponse

import cv2
import numpy as np
import json

fd = FaceDetector()

@csrf_exempt
def frame_detect(request):
    """
    Runs facial detection on a frame that is sent via a REST
    API call
    """
    # print(fd.detect_faces(cv2.imdecode(request.body, 1)))
    image = np.array(
        json.loads(request.body.decode('utf-8'))['frame'])
    image = cv2.imdecode(image, 1)
    detect = fd.detect_faces(image)
    return HttpResponse(detect)
