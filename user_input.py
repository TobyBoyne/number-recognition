import torch

import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter


def get_brush_mask(x, y, r=5):
	"""Get a circular brush as a numpy array
	"""
	X, Y = np.meshgrid(np.arange(28), np.arange(28))
	dists = (X - x)**2 + (Y - y)**2
	brush = 1/(1 + dists**2)
	return brush

class DrawUI:
	def __init__(self, size=28, key_fn=None, network=None):
		self.size = size
		self.drawing = np.zeros((size, size))
		self.fig, ax = plt.subplots()
		self.fig.canvas.mpl_connect('motion_notify_event', self.draw)
		self.fig.canvas.mpl_connect('button_press_event', self.clear)
		if key_fn and network:
			self.fig.canvas.mpl_connect('key_press_event', key_fn(self, network))

		self.image = ax.imshow(self.drawing, cmap="gray", vmin=0, vmax=1)


		timer = self.fig.canvas.new_timer(interval=50)
		timer.add_callback(self.update, ax)
		timer.start()
		plt.show()

	def add_space_fn(self, func):
		self.fig.canvas.mpl_connect('key_press_event', func)


	def draw(self, event):
		if event.button == 1:  # left-click
			t0 = perf_counter()
			x, y = event.xdata, event.ydata
			brush_mask = get_brush_mask(x, y)
			self.drawing = np.maximum(brush_mask, self.drawing)
			print(perf_counter() - t0)

	def clear(self, event):
		if event.dblclick:
			self.drawing = np.zeros((self.size, self.size))

	def update(self, ax):
		self.image.set_data(self.drawing)
		self.fig.canvas.draw()

if __name__ == "__main__":
	draw = DrawUI()