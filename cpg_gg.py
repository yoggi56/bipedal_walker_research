# CPG-based Gait Generator for OpenAI BipedalWalker-v3
#
# This algorithm produces trajectories for robot's foot endpoints in order to locomote.
# It is only trajectory generator without any feedbacks.
#
# All the neccesary information about this algorithm see here 
# https://www.researchgate.net/publication/362930025_CPG-Based_Gait_Generator_for_a_Quadruped_Robot_with_Sidewalk_and_Turning_Operations

from math import *
from scipy.integrate import ode

LEG_LENGTH = 34.0/30.0

class CPGGaitGenerator(object):
    def __init__(self):
        self.x = [0]*2
        self.y = [0]*2
        self.r = [0]*2
        self.w = [0]*2
        self.sum_con = [[0,0],[0,0]]
        self.w_sw = 1*pi
        self.beta = 2
        self.a = 100
        self.lamb = 8
        self.phi = [pi, 0]
        self.alpha = 100
        self.mu = 1  
        self.xy_init = [0.0, 0.0, 0.01, 0.01]
        self.t0 = 0.0
        self.Sh = 0.08
        self.Sl = 0.08
        self.P_B3 = {1: {"x": 0, "z": -1.8},
                    2: {"x": 0, "z": -1.8},
                    }

    def init_integrator(self, dt):
        self.de = ode(self.hopf_osc)
        self.de.set_integrator('dopri5')
        self.de.set_initial_value(self.xy_init, self.t0)
        self.dt = dt

    def set_static_params(self, ampl=1, alpha=0.25, lamb=1, a=100):
        self.mu = sqrt(ampl)
        self.alpha = alpha
        self.lamb = lamb
        self.a = a

    # w_sw - step frequency
    # beta - duty factor
    # Sh - step height
    # Sl - step length
    def set_gait_params(self, w_sw, beta, Sh, Sl):
        self.w_sw = w_sw
        self.beta = beta
        self.Sh = Sh
        self.Sl = Sl

    # differential equations for Hopf oscillator
    def hopf_osc(self, t, xy):
        self.x[0], self.y[0], self.x[1], self.y[1] = xy

        for i in range(2):
            self.r[i] = sqrt(self.x[i]**2 + self.y[i]**2)
            self.w[i] = self.w_sw*((self.beta/(exp(-self.a*self.y[i])+1)) + (1/(exp(self.a*self.y[i])+1)))
            self.sum_con[i][0] = 0
            self.sum_con[i][1] = 0
            for j in range(2):
                self.sum_con[i][0] += self.lamb*(cos(self.phi[i]-self.phi[j])*self.x[j] - sin(self.phi[i]-self.phi[j])*self.y[j])
                self.sum_con[i][1] += self.lamb*(sin(self.phi[i]-self.phi[j])*self.x[j] + cos(self.phi[i]-self.phi[j])*self.y[j])
        
        return [self.alpha*(self.mu-self.r[0]**2)*self.x[0]-self.w[0]*self.y[0]+self.sum_con[0][0], self.alpha*(self.mu-self.r[0]**2)*self.y[0]+self.w[0]*self.x[0]+self.sum_con[0][1],
                self.alpha*(self.mu-self.r[1]**2)*self.x[1]-self.w[1]*self.y[1]+self.sum_con[1][0], self.alpha*(self.mu-self.r[1]**2)*self.y[1]+self.w[1]*self.x[1]+self.sum_con[1][1]]
    
    # Maps phase trajectories into leg endpoints
    # phi_out - phase value from Hopf oacillator
    # leg_number - robot's leg number
    def mapping_function(self, phi_out, leg_number=1):
        # Stance phase
        if -pi <= phi_out < 0:
            Px = -((1/pi)*self.Sl*phi_out + self.Sl/2) + self.P_B3[leg_number]["x"]
            Pz = self.P_B3[leg_number]["z"]
        # Swing Phase
        elif 0 <= phi_out <= pi:
            Px = -((1/(2*pi))*self.Sl*sin(2*phi_out) - (1/pi)*self.Sl*phi_out + self.Sl/2) + self.P_B3[leg_number]["x"]
            if 0 <= phi_out < pi/2:
                Pz = (-1/(2*pi))*self.Sh*sin(4*phi_out) + (2/pi)*self.Sh*phi_out + self.P_B3[leg_number]["z"]
            else:
                Pz = (1/(2*pi))*self.Sh*sin(4*phi_out) - (2/pi)*self.Sh*phi_out + 2*self.Sh + self.P_B3[leg_number]["z"]

        return [Px, Pz]

    # Inverse Kinematics problem solver
    # Input:
    # pr - [list(2)] left leg coordinates x, y
    # pl - [list(2)] right leg coordinates x, y
    # Output:
    # theta1_r - right leg hip joint
    # theta2_r - right leg knee joint
    # theta1_l - left leg hip joint
    # theta2_l - left leg knee joint
    def ikine(self, pr, pl):
        l1 = LEG_LENGTH
        #leg1
        l2_r = sqrt(pr[0]**2 + pr[1]**2)
        theta12_r = acos(l2_r/(2*l1))
        theta1_r = (atan2(pr[0], abs(pr[1]))) + theta12_r
        theta2_r = -2*theta12_r
        
        #leg2
        l2_l = sqrt(pl[0]**2 + pl[1]**2)
        theta12_l = acos(l2_l/(2*l1))
        theta1_l = (atan2(pl[0], abs(pl[1]))) + theta12_l
        theta2_l = -2*theta12_l

        return [theta1_r, theta2_r, theta1_l, theta2_l]
    
    # form next trajectory coordinate
    # Output:
    # P_RF - desired right foot coordinates [x, y]
    # P_LF - desired left foot coordinates [x, y]
    def step(self):
        self.de.integrate(self.de.t + self.dt)

        self.phi_out1 = atan2(self.de.y[1], self.de.y[0])
        self.phi_out2 = atan2(self.de.y[3], self.de.y[2])

        P_RF = self.mapping_function(self.phi_out1, 1)
        P_LF = self.mapping_function(self.phi_out2, 2)

        return [P_RF, P_LF]

