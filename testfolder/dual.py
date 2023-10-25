import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
#from thumangle import Thumangle

max_num_hands=2

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
           64:'1200',65:'?'}
rps_gesture=gesture

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9)

# Gesture recognition model
file = np.genfromtxt('only_onemore_angle_fy_6500_20_beforseperate_2.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# neighbours 리스트에서 각 값들의 확률 계산 함수
def calculate_probabilities(neighbours):
    total = 0  # 주변 값의 총 개수
    probabilities = {}  # 각 값의 확률을 저장할 딕셔너리
    
    for inner_arr in neighbours:
        for value in inner_arr:
            if value in probabilities:
                probabilities[value] += 1
            else:
                probabilities[value] = 1
            total += 1
    
    for value in probabilities:
        probabilities[value] /= total  # 값들의 확률 계산
    probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    return probabilities

action_seq=[]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
#수정후
            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,2,2,0,0,0,0,2],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,3,5,8,12,16,20,17],:] # Child joint
            v = v2 - v1 # [26,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,22,23,24,20],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,23,24,25,26],:])) # [20,]
            angle = np.degrees(angle) # Convert radian to degree
            
            # thumangle = Thumangle() 
            # thumb_add_angle = thumangle.calculate_thumb_angle(joint)
            # angle = np.append(angle, thumb_add_angle)
            
                
            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            
            
            # neighbours에 있는 값들의 확률 딕셔너리 생성
            probabilities = calculate_probabilities(neighbours)
            idx = int(results[0][0]) #idx를 knn예측값으로 초기화
           
            #초기화한 값이 옳을 확률이 크지만 엄지를 고려한 추가 분류모델을 돌려보기.                
            action = idx #현재동작
            action_seq.append(action)
            if len(action_seq) < 3:
                continue
            
            this_action = 65 #모르는 동작을 ?로 처리(굳이 이거 필요없긴함)
            if action_seq[-1] == action_seq[-2]: #갑자기 튄 액션에 대해서 이전 액션으로 취급하는 코드
                this_action = action
            else:
                this_action=action_seq[-3]

            # Draw gesture result
            
            org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
            cv2.putText(img, text=rps_gesture[this_action].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255, 255), thickness=2)
                
            #test를 위한 코드
            print(results)
            print(neighbours)    
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)


    cv2.imshow('Hand type', img)
    if cv2.waitKey(1) == ord('q'):
        break