import cv2 as cv
import numpy as np
camera= cv.Videocapture(0)

while True:
  _, frame = camera.read()
  cv.imshow('Camera', frame)
  laplacian = cv.Laplacian(frame, cv.CV_64F)
  laplacian = np.uint8(laplacian)
  cv.imshow("Laplacian", laplacian)

  edges = cv.Canny(frame,100,100)
  cv.imshow('Canny', edges)
  
  if cv.waitKey(S) == ord('x'):
    break
camera.release()
cv.destroyAllWindows()
