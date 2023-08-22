import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray , frame):
    faces = face_cascade.detectMultiScale(gray , 1.3 , 5)
    for (x , y , w , h) in faces :
        cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 2)
        detect_face = gray[y:y+h , x:x+w]                          
        detect_colored_face = frame[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(detect_face , 1.1 , 3 )
        for (a , b , c , d) in eyes:
            cv2.rectangle(detect_colored_face , (a , b) , (a + c , b + d) , (255 , 0 , 0) , 2)
            
        smiles = smile_cascade.detectMultiScale(detect_colored_face, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(detect_colored_face, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    return frame

 

video_capture = cv2.VideoCapture(0)

while True:
    ret , frame = video_capture.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    my_output = detect(gray , frame)
    cv2.imshow('zende baad' , my_output)
    
    if cv2.waitKey(14) & 0XFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()

        