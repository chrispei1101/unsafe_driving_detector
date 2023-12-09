import dlib
import cv2
import numpy as np

head_not_forward_count = 0
nod_count = 0
face_detected = False


def landmarks_to_np(landmarks, dtype="int"): #work cited: TianjingWu https://github.com/TianxingWu for detecting faces
    num = landmarks.num_parts
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0) # connect to your camera
queue = np.zeros(30,dtype=int)
queue = queue.tolist()

while(cap.isOpened()):
    face_detected = False
    # read frame
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # for every face
    for i, rect in enumerate(rects):
        face_detected = True
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        # draw boxes
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, "Face {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # get landmarks        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
     
        # eye distance
        d1 =  np.linalg.norm(landmarks[37]-landmarks[41])
        d2 =  np.linalg.norm(landmarks[38]-landmarks[40])
        d3 =  np.linalg.norm(landmarks[43]-landmarks[47])
        d4 =  np.linalg.norm(landmarks[44]-landmarks[46])
        d_mean = (d1+d2+d3+d4)/4
        d5 =np.linalg.norm(landmarks[36]-landmarks[39])
        d6 =np.linalg.norm(landmarks[42]-landmarks[45])
        d_reference = (d5+d6)/2
        d_judge = d_mean/d_reference
        
        flag = int(d_judge<0.2) #0.2 is the threshod, adjust bigger if you have larger eyes unlike me

        queue = queue[1:len(queue)] + [flag] #enqueue flag
        
        # if more than half surpass threshod
        if sum(queue) > len(queue)/2 :
            cv2.putText(img, "Driver Eye Closed !", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)   
                
        # Calculate the face slope to determine if facing forward (haven't inplemented)
        face_slope = (landmarks[8][1] - landmarks[27][1]) / (landmarks[8][0] - landmarks[27][0])

        mouth_open_threshold = 10  # Adjust the threshold as needed
        if landmarks[66][1] - landmarks[62][1] > mouth_open_threshold:
            yawn_count += 1
            if yawn_count >= 10: #if surpass 10 frames
                cv2.putText(img, "Driver Yawning!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            yawn_count = 0

        if head_not_forward_count == 0 and yawn_count == 0 and sum(queue) <= len(queue)/2:
            cv2.putText(img, "SAFE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
    if not face_detected:
        cv2.putText(img, "Face Not Facing Forward!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("Camera", img)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):  # 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
