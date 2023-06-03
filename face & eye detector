import cv2
import keyboard

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
    
prev_state = False
curr_state = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    prev_state = curr_state
    curr_state = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 10, 255), 6)

        cropped_face = frame[y:y + h, x:x + w]
        cropped_face_gray = gray[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(cropped_face_gray, 1.1, 5)

        for (ex, ey, ew, eh) in eyes:
            curr_state = True
            cv2.rectangle(cropped_face, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

        if prev_state == False and curr_state == True:
            keyboard.send('command+t')  # resume

        if prev_state == True and curr_state == False:
            keyboard.send('command+p')  # pause

    cv2.imshow('Face detection', cv2.resize(frame, (1000, 700)))
    #cv2.imshow('Face detection', frame)

    k = cv2.waitKey(10)

    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
