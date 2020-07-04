import torch

import matplotlib.pyplot as plt

from testing import load_model
from user_input import DrawUI

def solve_event_fn(draw_ui, network):
	def f(event):
		inputs = torch.from_numpy(draw_ui.drawing)
		inputs = inputs.unsqueeze(0)
		inputs = inputs.unsqueeze(0).float()
		outputs = network.forward(inputs)
		n = torch.argmax(outputs)
		print(n.item())
	return f


if __name__ == "__main__":
	net = load_model()
	draw = DrawUI(key_fn=solve_event_fn, network=net)