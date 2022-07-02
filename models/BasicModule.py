#-*- encoding:utf-8 -*-
import random

import torch
import torch as t
import time
import numpy as np

seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))
        
    def load(self,path):

        self.load_state_dict(t.load(path))
        
    def save(self,name=None):

        
        if name is None:
            prefix = ""
            name = time.strftime("%y%m%d_%H:%M:%S.pth".format(prefix))
            
        t.save(self.state_dict(),name)
        return name
