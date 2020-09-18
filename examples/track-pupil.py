import os
import sys
import time

import cv2
import eyerec


def track_pupil(method, video_file):
    tracker = eyerec.PupilTracker(name=method)
    cap = cv2.VideoCapture(video_file)
    sys.stdout.write(f"x, y, width, height, angle, confidence, runtime_ms,{os.linesep}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

        start = time.perf_counter()
        pupil = tracker.detect(timestamp, gray)
        end = time.perf_counter()
        runtime_ms = 1e3 * (end - start)
        for output in [
            pupil['center'][0],
            pupil['center'][1],
            pupil['size'][0],
            pupil['size'][1],
            pupil['angle'],
            pupil['confidence'],
            runtime_ms,
            ]:
            sys.stdout.write(f"{output:.4f}, ")
        sys.stdout.write(os.linesep)

        if pupil['confidence'] > 0.66:
            g = 255 * pupil['confidence']
            r = 255 - g
            cv2.ellipse(frame, (pupil['center'], pupil['size'], pupil['angle']), (0, g, r), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    method = sys.argv[1]
    video_file = sys.argv[2]
    track_pupil(method=method, video_file=video_file)
