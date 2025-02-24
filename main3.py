import cv2 
import mediapipe as mp 
import numpy as np

mpHands=mp.solutions.hands
hands=mpHands.Hands(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                    max_num_hands=1)
draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
canvas=np.zeros((480,640,3),dtype=np.uint8)
prev_x,prev_y=0,0

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
       break
    frame=cv2.flip(frame,1)
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    processed=hands.process(frameRGB)
    if processed.multi_hand_landmarks:
      for handlm in processed.multi_hand_landmarks:
        h,w,_=frame.shape
        index_finger_tip=handlm.landmark[8]
        x,y=int(index_finger_tip.x*w),int(index_finger_tip.y*h)
        if prev_x!=0 and prev_y!=0:
           cv2.line(canvas,(prev_x,prev_y),(x,y),(0,255,0),10)
        prev_x,prev_y=x,y
        draw.draw_landmarks(frame,handlm,mpHands.HAND_CONNECTIONS)
    frame=cv2.addWeighted(frame,0.5,canvas,0.5,0)
    cv2.imshow('air paint',frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('c'):
      canvas=np.zeros((480,640,3),dtype=np.uint8)
    elif key==ord('q'):
      break
cap.release()    
cv2.destroyAllWindows()
