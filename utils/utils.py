import torch
import numpy as np
import random

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

_image_net_classes = [0, 217, 482, 491, 497]

def target_transforms(y):
  # Assuming your dataset has target values in the range 0-999
  if y < len(_image_net_classes):
    return _image_net_classes[y]
  else:
    return y  # Return the original target if it's outside the range of _image_net_classes
