import torch

import matplotlib.pyplot as plt

from testing import load_model
from user_input import DrawUI

def solve_event_fn(draw_ui, network):
	def f(x):
		inputs = torch.from_numpy(draw_ui.drawing)
		inputs = inputs.unsqueeze(0)
		inputs = inputs.unsqueeze(0).float()
		outputs = network.forward(inputs)
		n = torch.argmax(outputs)
		print(n)
	return f


if __name__ == "__main__":
	net = load_model()
	draw = DrawUI()

	f = solve_event_fn(draw, net)
	draw.fig.canvas.mpl_connect('key_press_event', f)

	plt.show()