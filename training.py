import torchvision

dataset = torchvision.datasets.MNIST("training-data", train=True, download=True)
print(dataset)