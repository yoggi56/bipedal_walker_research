

from cpg_gg import CPGGaitGenerator
from math import *
import numpy as np
from gym import spaces
from my_bipedal_walker import BipedalWalker

# If you want to train only regular version of bipedalWalker-v3
# can uncomment standard library:
# from gym.envs.box2d.bipedal_walker import BipedalWalker


SIM_FREQ = 50
KP = 1
KD = 0.1
LEG_LENGTH = 34.0/30.0
X_MAX = LEG_LENGTH * sqrt(2)

class BipedalWalkerPMTG(BipedalWalker):
    hardcore = False

    def __init__(self, is_hard=False, action_repeat=3, act_noise=0.3, rew_scale=5.0, learn=True, episode_length=2000):
        super().__init__(learn=learn)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1, -1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32),
        )

        if is_hard == True:
            self.hardcore = True
        else:
            self.hardcore = False

        self.Hg = 0.5
        self.Ls = 1.6
        self.beta = 2
        self.w_sw = pi
        self.dt = 1/SIM_FREQ # direiative time

        self.cpg = CPGGaitGenerator()
        self.cpg.set_static_params()
        self.cpg.set_gait_params(w_sw=self.w_sw, beta=self.beta, Sh=self.Hg, Sl=self.Ls)
        self.cpg.init_integrator(self.dt)

        self.action_std = [0]*4
        self.torque = [0]*4
        self.obs = [0]*25

        # some tricks to enhance performance
        self.action_repeat = action_repeat
        self.act_noise = act_noise
        self.reward_scale = rew_scale
        self.it = 0
        self.episode_length = episode_length

    def reset(self):
        self.cpg.set_gait_params(w_sw=self.w_sw, beta=self.beta, Sh=self.Hg, Sl=self.Ls)
        self.cpg.init_integrator(self.dt)

        self.resetting = True
        self.obs = super().reset()
        self.resetting = False

        self.it = 0
        
        return self.obs
        
    def set_ref_vel(self, ref_vel):
        return super().set_ref_vel(ref_vel)

    def step(self, action):
        if self.resetting != True:
            self.it += 1
            # input to CPG block
            self.cpg.set_gait_params(w_sw=action[0], beta=action[1], Sh=action[2], Sl=action[3])
            # CPG output
            point_ref = self.cpg.step()
            # Sum the agent and cpg outputs
            point_ref[0][0] = point_ref[0][0] + action[4]
            point_ref[0][1] = point_ref[0][1] + action[5]
            point_ref[1][0] = point_ref[1][0] + action[6]
            point_ref[1][1] = point_ref[1][1] + action[7]
            # inverse kinematics
            theta_ref = self.ikine(point_ref[0], point_ref[1])
            # PD-regulator
            cur_angles = [self.obs[4], self.obs[6], self.obs[9], self.obs[11]]
            cur_speed = [self.obs[5], self.obs[7], self.obs[10], self.obs[12]]
            self.torque[0] = self.pd_regulator(theta_ref[0], cur_angles[0], cur_speed[0])
            self.torque[1] = self.pd_regulator(theta_ref[1]+1, cur_angles[1], cur_speed[1])
            self.torque[2] = self.pd_regulator(theta_ref[2], cur_angles[2], cur_speed[2])
            self.torque[3] = self.pd_regulator(theta_ref[3]+1, cur_angles[3], cur_speed[3])
            self.action_std = self.torque

            self.action_std += self.act_noise * (-2 * np.random.random(4) + 1)
            r = 0.0
            for _ in range(self.action_repeat):
                self.obs, rewards, dones, info = super().step(self.action_std)
                if self.it >= self.episode_length:
                    dones = 1
                if dones and rewards == -100:
                    rewards = 0
                r = r + rewards
                if dones and self.action_repeat != 1:
                    return self.obs, 0.0, dones, info
                if self.action_repeat == 1:
                    return self.obs, r, dones, info
            return self.obs, self.reward_scale*r, dones, info
            
        else:
            self.obs, rewards, dones, info = super().step(action)
            return self.obs, rewards, dones, info

    def pd_regulator(self, ref_angle, cur_angle, cur_speed):
        e = ref_angle - cur_angle
        torque = KP*e - KD*cur_speed
        return torque

    def ikine(self, pr, pl):
        l1 = LEG_LENGTH
        # check reachable region
        if pr[0] > X_MAX:
            pr[0] = X_MAX
        if pr[1] > X_MAX:
            pr[1] = X_MAX
        if pl[0] > X_MAX:
            pl[0] = X_MAX
        if pl[1] > X_MAX:
            pl[1] = X_MAX

        if pr[0] < -X_MAX:
            pr[0] = -X_MAX
        if pr[1] < -X_MAX:
            pr[1] = -X_MAX
        if pl[0] < -X_MAX:
            pl[0] = -X_MAX
        if pl[1] < -X_MAX:
            pl[1] = -X_MAX
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

