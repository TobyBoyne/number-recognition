import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from network import Net

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

transform = transforms.Compose(
	[transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST("./training-data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										  shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST("./training-data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=2)



if __name__ == "__main__":
	# data_iter = iter(testloader)
	# # data_iter.next() takes next batch (size=4)
	# images, labels = data_iter.next()
	# print(labels)
	# imshow(torchvision.utils.make_grid(images, padding=0))

	# create new network, optimizer, and loss criterion
	net = Net()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
	loss_fn = torch.nn.MSELoss()

	# training loop
	for epoch in range(2):

		running_loss = 0.0
		for i, data in enumerate(trainloader):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			target = net.get_target(labels)
			loss = loss_fn(outputs, target)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:  # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')
