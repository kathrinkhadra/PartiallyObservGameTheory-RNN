import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import math

class Kalman:

    def __init__(self,dt,pos_init,kf,v_E,p_pos): #dt,None,None,None,[]
        self.dt=dt
        self.pos_init=pos_init
        self.kf=kf
        self.v_E=v_E
        self.p_pos=p_pos

    def fx(self, x, dt):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])

        return np.dot(F, x)

    def hx(self, x):
        #x=[x_pos,vel_x,y_pos,vel_y]
        #try only with THETA P HERE!!

        #vel_x=self.v_E*np.cos(x[0])
        #vel_y=self.v_E*np.sin(x[1])

       # measurement function - convert state into a measurement

       return x[[0,2]]#,vel_x,vel_y,

    def build_Kalman_EKF(self):

       self.kf = KalmanFilter(dim_x=4, dim_z=2)

       self.kf.x = self.pos_init

       self.kf.F = np.array([[1, self.dt, 0,  0],
                 [0,  1, 0,  0],
                 [0,  0, 1, self.dt],
                 [0,  0, 0,  1]])
       self.kf.H =np.array([[1, 0,0,0], [0, 0,0,1]])
       self.kf.P *=0.2
       z_std = 0.09
       self.kf.R = np.diag([z_std**2, z_std**2])
       self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
       self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

    def build_Kalman(self):
        # create sigma points to use in the filter
        points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=1.)

        #self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=self.fx, hx=self.hx, points=points)
        self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=self.fx, hx=self.hx, points=points)
        self.kf.x = self.pos_init # initial state
        self.kf.P *= 0.2 # initial uncertainty
        z_std = 0.09
        self.kf.R =np.diag([z_std**2, z_std**2]) # 1 standard
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)
