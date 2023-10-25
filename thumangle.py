import numpy as np
from decimal import Decimal


class Thumangle:
    def __init__(self):
        pass
        
    def calculate_point_coordinate(self, joints):
        point = np.zeros((6, 3))  #순서대로 P0,P5,P17,P4,H',O를 정의
        point[0] = joints[0]
        point[1] = joints[5]
        point[2] = joints[17]
        point[3] = joints[4]
        
        t=(np.sum(point[3])-np.sum(point[0]))/(np.square(np.linalg.norm(point[0] - point[1])))
        k=(np.sum(point[2])-np.sum(point[0]))/(np.square(np.linalg.norm(point[0] - point[1])))
        point[4]=t*(point[1]-point[0])+point[0]
        point[5]=k*(point[1]-point[0])+point[0]
            
        return point
    
    def calculate_vector(self, joints):
        point=self.calculate_point_coordinate(joints)
        vector = np.zeros((2, 3))  #순서대로 벡터(P17->O),벡터(H'->P4)를 정의
        vector[0]=point[5]-point[2]
        vector[1]=point[3]-point[4] 
        
        return vector
    
    def calculate_thumb_angle(self, joints):
        vector=self.calculate_vector(joints)
        
        # 벡터의 크기 계산
        norm_v0 = np.linalg.norm(vector[0])
        norm_v1 = np.linalg.norm(vector[1])
        cos_angle = (np.dot(vector[0],vector[1]))/((norm_v0*norm_v1))
        angle_rad=np.arccos(cos_angle)
        
        # 음수인 경우 각도가 직각보다 큰 값으로 변환
        angle_rad = np.where(angle_rad< 0, 2 * np.pi - angle_rad, angle_rad)
        angle_deg=np.degrees(angle_rad)
        return angle_deg