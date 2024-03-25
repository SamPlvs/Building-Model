import torch
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import os
import matplotlib.pyplot as plt
import numpy as np 

def load_data(name='mnist', batch_size=4):
	transform = transforms.Compose([transforms.ToTensor()])

	batch_size = batch_size
	root = os.path.join(os.getcwd(), 'data')
	print('saving data to: {}'.format(root))

	if name=='mnist':
		print('====== Loading MNIST ======')
		trainset = datasets.MNIST(root, train=True, download=True, transform=transform)
		testset = datasets.MNIST(root, train=False, download=True, transform=transform) 
		classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

	else:
		print('====== Loading CIFAR10 ======')
		trainset = datasets.CIFAR10(root, train=True, download=True, transform=transform)
		testset = datasets.CIFAR10(root, train=False, download=True, transform=transform)

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	print('no. of training samples: {}, no. of testing_samples: {}'.format(len(trainset), len(testset)))

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)

	return trainloader, testloader, classes


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(model, optimizer, criterion, epochs, trainloader):
	
	for epoch in range(epochs):  # loop over the dataset multiple times
	    running_loss = 0.0

	    for i, data in enumerate(trainloader, 0):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs, labels = data

	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        outputs = model(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
	            running_loss = 0.0

	print('Finished Training')

	return model, optimizer


def test(model, testloader):
	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        # calculate outputs by running images through the network
	        outputs = model(images)
	        # the class with the highest energy is what we choose as prediction
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def get_class_performance_breakdown(model, testloader, classes):
	# prepare to count predictions for each class
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}

	# again no gradients needed
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = model(images)
	        _, predictions = torch.max(outputs, 1)
	        # collect the correct predictions for each class
	        for label, prediction in zip(labels, predictions):
	            if label == prediction:
	                correct_pred[classes[label]] += 1
	            total_pred[classes[label]] += 1


	# print accuracy for each class
	for classname, correct_count in correct_pred.items():
	    accuracy = 100 * float(correct_count) / total_pred[classname]
	    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')








