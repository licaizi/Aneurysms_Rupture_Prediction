import torch.nn as nn

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 0.00001
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
		denom = (input_flat * input_flat).sum(1) + (target_flat * target_flat).sum(1)
 
		loss = 2 * (intersection.sum(1)) / (denom + smooth)
		loss = 1 - loss.sum() / N
 
		return loss