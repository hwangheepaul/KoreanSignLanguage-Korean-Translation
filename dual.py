import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from thumangle import Thumangle
##import win32api

# 현재 모니터의 해상도 가져오기
##screen_res = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

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
file = np.genfromtxt('test_add_data.csv', delimiter=',')
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

# 손의 위치를 판단하는 함수
def determine_hand_location(y_coord, avg_eyes_y, avg_shoulders_y, avg_joints_y):
    if y_coord is None:
        return 3
    if y_coord < avg_eyes_y:
        return 0
    elif y_coord < avg_shoulders_y:
        return 1
    elif y_coord < avg_joints_y:
        return 2
    else:
        return 3

#pose detection 함수 생성
def detect_pose_landmarks(img, pose_model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose_model.process(img_rgb)
    hand_location_right = None  # 오른손 위치 초기화
    hand_location_left = None  # 왼손 위치 초기화

    if result.pose_landmarks:
        # 파란색으로 선을 그리기 위한 DrawingSpec 객체 생성
        blue_color = (255, 0, 0)  # BGR 형식의 파란색
        drawing_spec = mp_drawing.DrawingSpec(color=blue_color, thickness=2)
        
        left_index = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value]
        right_index = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value]

        # 중앙 두 눈의 평균 y좌표
        left_eye_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y
        right_eye_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y
        avg_eyes_y = (left_eye_y + right_eye_y) / 2

        # 어깨선 평균 y좌표
        left_shoulder_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        avg_shoulders_y = (left_shoulder_y + right_shoulder_y) / 2

        # 두 팔꿈치와 두 엉덩이의 y좌표 평균
        left_elbow_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        right_elbow_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        left_hip_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        avg_joints_y = (left_elbow_y + right_elbow_y + left_hip_y + right_hip_y) / 4

        # 콘솔창에 출력
        #print(f"중앙 두 눈의 평균 y좌표: {avg_eyes_y:.3f}")
        #ㅎprint(f"어깨선 평균 y좌표: {avg_shoulders_y:.3f}")
        #print(f"두 팔꿈치와 두 엉덩이의 y좌표 평균: {avg_joints_y:.3f}")

        hand_location_right = determine_hand_location(right_index.y, avg_eyes_y, avg_shoulders_y, avg_joints_y)
        hand_location_left = determine_hand_location(left_index.y, avg_eyes_y, avg_shoulders_y, avg_joints_y)

        # 콘솔창에 손의 위치 출력
        #print(f"Right Hand Location: {hand_location_right}")
        #print(f"Left Hand Location: {hand_location_left}")

        # 각 손의 손위를 화면에 출력
        cv2.putText(img, f"Location: {hand_location_right}", (int(right_index.x * img.shape[1]) + 50, int(right_index.y * img.shape[0]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, f"Location: {hand_location_left}", (int(left_index.x * img.shape[1]) + 50, int(left_index.y * img.shape[0]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # 화면상 손의 y좌표 출력 (좌우 반전을 고려)
        #cv2.putText(img, f"Right {left_index.y:.3f}", (int(left_index.x * img.shape[1]) + 50, int(left_index.y * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        #cv2.putText(img, f"Left {right_index.y:.3f}", (int(right_index.x * img.shape[1]) + 50, int(right_index.y * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)

    return img, hand_location_right, hand_location_left
        
# MediaPipe Pose model 초기화
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    left_hand_detected = False  # 왼손 감지 여부를 나타내는 변수
    right_hand_detected = False  # 오른손 감지 여부를 나타내는 변수
    hand_location_left = None
    hand_location_right = None

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
             # 왼손과 오른손의 landmark를 사용하여 어느 쪽 손인지 판별
            left_landmark = res.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_landmark = res.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            if left_landmark.x < right_landmark.x:
                left_hand_detected = True
                print("Left hand detected: ", rps_gesture[this_action], " Hand Location: ", hand_location_left)
                
            else:
                right_hand_detected = True
                print("Right hand detected: ", rps_gesture[this_action], " Hand Location: ", hand_location_right)

            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,2,2,0,0,0,0,2],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,3,5,8,12,16,20,17],:] # Child joint
            v = v2 - v1 # [26,3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # Normalize v

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
            # print("results:",results)
            #print("neighbours:",neighbours)    
            
            # 손 위치를 pose 모델로 감지하고 이미지에 그림
            img, hand_location_right, hand_location_left = detect_pose_landmarks(img, pose_model)

            # 왼손 감지 시 동작과 함께 왼손 위치 출력
            if left_hand_detected:
                print("Left hand detected: ", rps_gesture[this_action], " Hand Location: ", hand_location_left)

            # 오른손 감지 시 동작과 함께 오른손 위치 출력
            elif right_hand_detected:
                print("Right hand detected: ", rps_gesture[this_action], " Hand Location: ", hand_location_right)
          
    cv2.imshow('Hand type', img)
    if cv2.waitKey(1) == ord('q'):
        break
