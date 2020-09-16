import cv2
numberPlate = cv2.CascadeClassifier('CascadeAlgo/haarcascade_russian_plate_number.xml')
cap = cv2.VideoCapture('Data/nPlate.mp4')
img = cv2.imread('Data/np3.PNG')


while True:
    ret, frame = cap.read()

    detectPlate = numberPlate.detectMultiScale(frame)
    for (x, y, w, h) in detectPlate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roi_img = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi_img, (31, 31), 0)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
