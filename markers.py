import numpy as np

import cv2

cam = cv2.VideoCapture(2)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        if ids[0][0] == 2:
            coords = corners[0][0]
            # for i, c in enumerate(coords):
            #     cv2.circle(frame, (int(c[0]), int(c[1])), 10, (0, 255, 0), -1)
            #     cv2.putText(frame, str(i), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0, 1))
            M = cv2.getPerspectiveTransform(np.float32(coords), np.float32([[0, 0], [0, 1], [1, 1], [1, 0]]))
            R1, R2, t = cv2.decomposeEssentialMat(M)
            print()

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
