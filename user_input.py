import torch

import matplotlib.pyplot as plt
import numpy as np

a = np.array([3, 4])
b = np.array([9, 0])

print(np.maximum(a, b))


letter_drawing = np.zeros((28, 28))

def draw(event):
	if event.button == 1: # left-click
		x, y = event.xdata, event.ydata
		brush_mask = get_brush_mask(x, y)
		



def get_brush_mask(x, y, r=5):
	"""Get a circular brush as a numpy array
	(x, y) is the distance of the brush from the centre corner of a square
	r is the radius of the brush mask, such that the centre is at (r, r)
	"""
	X, Y = np.meshgrid(np.arange(28), np.arange(28))
	dists = (X - x)**2 + (Y - y)**2
	brush = 1/(1 + dists)
	# print(brush * 5)
	# plt.imshow(brush)
	# plt.show()
	return brush

def get_drawing_canvas():
	fig, ax = plt.subplots()
	fig.canvas.mpl_connect('motion_notify_event', draw)
	ax.imshow(letter_drawing, cmap="gray")
	plt.show()

get_drawing_canvas()