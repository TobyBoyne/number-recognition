import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	# img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

transform = transforms.Compose(
	[transforms.ToTensor()])
	#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST("./training-data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										  shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST("./training-data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=2)



# print images
if __name__ == "__main__":
	print(type(testloader))
	data_iter = iter(testloader)
	# data_iter.next() takes next batch (size=4)
	images, labels = data_iter.next()
	imshow(torchvision.utils.make_grid(images))
	# for i, element in enumerate(data_iter):
	# 	if i == 4: break
	# 	imshow(element)