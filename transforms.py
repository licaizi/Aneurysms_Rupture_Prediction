import random
import torch
import numpy as np
from torchvision.transforms import functional as F



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            axis = random.randint(1,2)
            image = torch.flip(image,[axis])
        return image



class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image).float()
        # image = F.to_tensor(image)
        return image