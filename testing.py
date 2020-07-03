import torch
import torchvision

from tqdm import tqdm

from network import Net
from training import PATH, transform


def load_model():
	net = Net()
	net.load_state_dict(torch.load(PATH))
	return net

testset = torchvision.datasets.MNIST("./training-data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=2)

if __name__ == "__main__":
	net = load_model()

	correct = 0
	total = 0
	with torch.no_grad():
		with tqdm(testloader) as tqdm_iterator:
			for data in tqdm_iterator:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

	print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')