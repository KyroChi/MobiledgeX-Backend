from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from tracker.serializers import DetectedSerializer
from facial_detection.facedetector import FaceDetector
from django.http import HttpResponse

import ast
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
    if request.method == 'POST':
        image = json.loads(request.body)['file']
        image = ast.literal_eval(image)
        image = np.array(image).astype(np.uint8)
        rects = fd.detect_faces(image)
        return HttpResponse(rects)
    
    return HttpResponse("Must send frame as a POST")
