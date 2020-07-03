import torch

import matplotlib.pyplot as plt
import numpy as np

def get_brush_mask(x, y, r=5):
	"""Get a circular brush as a numpy array
	"""
	X, Y = np.meshgrid(np.arange(28), np.arange(28))
	dists = (X - x)**2 + (Y - y)**2
	brush = 1/(1 + dists**2)
	return brush

class DrawUI:
	def __init__(self, size=28):
		self.size = size
		self.drawing = np.zeros((size, size))
		self.fig, ax = plt.subplots()
		self.fig.canvas.mpl_connect('motion_notify_event', self.draw)

		self.image = ax.imshow(self.drawing, cmap="gray", vmin=0, vmax=1)
		plt.show()

	def draw(self, event):
		if event.button == 1:  # left-click
			x, y = event.xdata, event.ydata
			brush_mask = get_brush_mask(x, y)
			self.drawing = np.maximum(brush_mask, self.drawing)
			self.image.set_data(self.drawing)
			self.fig.canvas.draw()

if __name__ == "__main__":
	draw = DrawUI()