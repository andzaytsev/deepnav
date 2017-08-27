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
     _action(-20,   0,  0,  0, 0, 0, 0), # look_left
     _action( 20,   0,  0,  0, 0, 0, 0), # look_right
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
        self.obs_specs = ['RGB_INTERLACED', 'VEL.TRANS']

        self.lab = deepmind_lab.Lab(map_name, self.obs_specs, config={
            'fps': str(Config.FPS),
            'width': str(Config.IMAGE_WIDTH),
            'height': str(Config.IMAGE_HEIGHT)
            })

        self.reset()

    def reset(self):
        if not self.lab.reset():
            assert 'Error reseting lab environment'
        
    def is_running(self):
        return self.lab.is_running()

    def get_state(self, spec='RGB_INTERLACED'):
        obs = self.lab.observations()  # dict of Numpy arrays
        if spec not in obs:
            assert '%s specification not in observation'%spec
        return obs[spec]
    
    @staticmethod
    def get_num_actions():
        return len(GameManager.ACTION_LIST)

    def step(self, action):
        if action == -1:
            reward = self.lab.step(_action(0, 0, 0, 0, 0, 0, 0))
        else:
            reward = self.lab.step(GameManager.ACTION_LIST[action], num_steps=4) 
        
        return reward, self.is_running()
