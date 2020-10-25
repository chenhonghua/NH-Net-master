import numpy as np
import torch
import math


def log_write(file_out, str_out, show_info=True):
    file_out.write(str_out+'\n')
    file_out.flush()
    if show_info:
        print(str_out)



def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)
