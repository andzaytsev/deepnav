# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import deepmind_lab
import numpy as np
from Config import Config
import sys

def _action(*entries):
      return np.array(entries, dtype=np.intc)

class GameManager:

    ACTION_LIST = [
     _action(-1*int(Config.ROTATION),   0,  0,  0, 0, 0, 0), # look_left
     _action( int(Config.ROTATION),   0,  0,  0, 0, 0, 0), # look_right
     #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
     #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
     _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
     _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
     _action(  0,   0,  0,  1, 0, 0, 0), # forward
     _action(  0,   0,  0, -1, 0, 0, 0), # backward
     #_action(  0,   0,  0,  0, 1, 0, 0), # fire
     #_action(  0,   0,  0,  0, 0, 1, 0), # jump
     #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
    ]

    def __init__(self, map_name):
        self.map_name = map_name
        self.obs_specs = ['RGBD_INTERLACED', 'VEL.TRANS', 'VEL.ROT']

        self.lab = deepmind_lab.Lab(map_name, self.obs_specs, config={
            'fps': str(Config.FPS),
            'width': str(Config.IMAGE_WIDTH),
            'height': str(Config.IMAGE_HEIGHT)
            })

        self.prev_action = 0
        self.prev_reward = 0
        self.reset()

    def reset(self):
        self.prev_action = 0
        self.prev_reward = 0
        if not self.lab.reset():
            assert 'Error reseting lab environment'
        
    def is_running(self):
        return self.lab.is_running()

    def get_state(self):
        obs = self.lab.observations()  # dict of Numpy arrays
        image = obs['RGBD_INTERLACED']

        # create a low resolution (4x16) depth map from the 84x84 image
        depth_map = image[:,:,3]
        depth_map = depth_map[16:-16,:] # crop
        depth_map = depth_map[:,2:-2] # crop
        depth_map = depth_map[::13,::5] # subsample

        image = image[:,:,:3].astype(np.float32) / 255. #RGB

        # flatten array for later append
        image = image.flatten()
        depth_map = depth_map.flatten()

        # quantize depth (as per DeepMind paper)
        depth_map = np.power(depth_map/255., 10)
        depth_map = np.digitize(depth_map,
            [0,0.05,0.175,0.3,0.425,0.55,0.675,0.8,1.01])  # bins
        depth_map -= 1

        # velocity vectors
        vel_vec1 = obs['VEL.TRANS'] 
        vel_vec2 = obs['VEL.ROT']

        # combined state
        state = np.append(image, depth_map) 
        state = np.append(state, vel_vec1)
        state = np.append(state, vel_vec2)
        state = np.append(state, self.prev_action)
        state = np.append(state, self.prev_reward)

        return state
    
    @staticmethod
    def get_num_actions():
        return len(GameManager.ACTION_LIST)

    def step(self, action):
        if action == -1:  #NO-OP
            reward = self.lab.step(_action(0, 0, 0, 0, 0, 0, 0))
        else:
            reward = self.lab.step(GameManager.ACTION_LIST[action], num_steps=4)
            self.prev_action = action
            self.prev_reward = reward
        
        return reward, self.is_running()
