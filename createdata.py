import cv2
import mediapipe as mp
import numpy as np
import time, os
from thumangle import Thumangle

max_num_hands=1

gesture = {0:'0000',1:'0001',2:'0002',3:'0003',4:'0004',5:'0005',
           6:'0100',7:'0101',8:'0102',
           9:'0200',10:'0201',11:'0202',12:'0203',13:'0204',14:'0205',
           15:'0300',16:'0301',17:'0302',18:'0303',19:'0304',20:'0305',
           21:'0400',22:'0401',23:'0402',24:'0403',
           25:'0500',26:'0501',27:'0502',
           28:'0600',29:'0601',30:'0602',31:'0603',32:'0604',33:'0605',34:'0606',
           35:'0700',36:'0701',37:'0702',38:'0703',39:'0704',40:'0705',41:'0706',
           42:'0800',43:'0801',44:'0802',
           45:'0900',46:'0901',47:'0902',48:'0903',49:'0904',50:'0905',51:'0906',52:'0907',53:'0908',54:'0909',55:'0910',56:'0911',57:'0912',58:'0913',59:'0914',
           60:'1000',61:'1001',
           62:'1100',63:'1101',
           64:'1200'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

# Gesture recognition dat5
file = np.genfromtxt('only_onemore_angle.csv', delimiter=',')
print(file.shape)
number=0
cap = cv2.VideoCapture(0)

def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,2,2,0,0,0,0,2],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,3,5,8,12,16,20,17],:] # Child joint
            v = v2 - v1 # [26,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,22,23,24,20],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,23,24,25,26],:])) # [19,]

            angle = np.degrees(angle) # Convert radian to degree
            # #엄지 각도 추가
            # thumangle = Thumangle() 
            # thumb_add_angle = thumangle.calculate_thumb_angle(joint)
            # angle = np.append(angle, thumb_add_angle)
            # print(angle)
            
            data = np.array([angle], dtype=np.float32)
            data = np.append(data, number)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

np.savetxt('only_onemore_angle_fy.csv', file, delimiter=',')


# LSTM사용
# seq_length = 1
# secs_for_action = 4
# gesture = ['0000','0001','0002','0003','0004','0005',
#            '0100','0101','0102',
#            '0200','0201','0202','0203','0204','0205',
#            '0300','0301','0302','0303','0304','0305',
#            '0400','0401','0402','0403',
#            '0500','0501','0502',
#            '0600','0601','0602','0603','0604','0605','0606'
#            '0700','0701','0702','0703','0704','0705','0706'
#            '0800','0801','0802',
#            '0900','0901','0902','0903','0904','0905','0906','0907','0908','0909','0910','0911','0912','0913','0914',
#            '1000','1001',
#            '1100','1101',
#            '1200']
# # MediaPipe hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0)

# created_time = int(time.time())
# os.makedirs('dataset', exist_ok=True)

# while cap.isOpened():
#     for idx, action in enumerate(actions):
#         data = []

#         ret, img = cap.read()

#         img = cv2.flip(img, 1)

#         cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
#         cv2.imshow('img', img)
#         if cv2.waitKey(0) == 1:
#             pass

#         start_time = time.time()
#         while time.time() - start_time < secs_for_action:
#             ret, img = cap.read()

#             img = cv2.flip(img, 1)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             result = hands.process(img)
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#             if result.multi_hand_landmarks is not None:
#                 for res in result.multi_hand_landmarks:
#                     joint = np.zeros((21, 4))
#                     for j, lm in enumerate(res.landmark):
#                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

#                     # Compute angles between joints
#                     v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
#                     v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
#                     v = v2 - v1 # [20, 3]
#                     # Normalize v
#                     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#                     # Get angle using arcos of dot product
#                     angle = np.arccos(np.einsum('nt,nt->n',
#                         v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
#                         v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

#                     angle = np.degrees(angle) # Convert radian to degree

#                     angle_label = np.array([angle], dtype=np.float32)
#                     angle_label = np.append(angle_label, idx)

#                     d = np.concatenate([joint.flatten(), angle_label])

#                     data.append(d)

#                     mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

#             cv2.imshow('img', img)
#             if cv2.waitKey(1) == ord('q'):
#                 break

#         data = np.array(data)
#         print(action, data.shape)
#         np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

#         # Create sequence data
#         full_seq_data = []
#         for seq in range(len(data) - seq_length):
#             full_seq_data.append(data[seq:seq + seq_length])

#         full_seq_data = np.array(full_seq_data)
#         print(action, full_seq_data.shape)
#         np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
#     break