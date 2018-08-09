import sys
import time
import cv2 as cv
from tracker.tracking.meta import logging

class FaceDetector(object):
    """ """
    resize_width = 0
    all_faces = []
    face_cascade = None
    scale = 0
    resized_width = 320
    face_tempate = None
    face_roi = None
    face_pos = None
    
    def __init__(self, cascade_file_path='default.xml'):
        """ """
        self.face_cascade = cv.CascadeClassifier(
            cascade_file_path)
        
    def detect_faces(self, frame):
        """ 
        Return a face object that represents a detected 
        face.
        """
        width, height = frame.shape[:2]
        self.all_faces = self.face_cascade.detectMultiScale(
            frame, 1.1, 3, 0,
            (int(width / 5), int(width / 5)),
            (int(width * 2 / 3), int(width * 2 / 3)))

        if len(self.all_faces) == 0:
            return []

        self.all_faces[:,2:] += self.all_faces[:,:2]
        return self.all_faces

    def test(self):
        print("I have been called")

if __name__ == "__main__":
    fd = FaceDetector()
    
    capture = cv.VideoCapture(0)

    while True:
        _, frame = capture.read()

        rects = fd.detect_faces(frame)
            
        for x1, y1, x2, y2 in rects:
            cv.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
    
    
        

