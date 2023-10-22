# Import necessary libraries
import torch  # PyTorch library for tensor computations and deep learning
import torchvision  # Extension of PyTorch for computer vision tasks
import torchvision.transforms as transforms  # Tools for image transformations
import matplotlib.pyplot as plt  # Library for plotting and visualization
import numpy as np  # Library for numerical computations
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define a series of transformations to apply to the images:
# 1. Convert images to PyTorch tensors.
# 2. Normalize the images so that their pixel values are between -1 and 1.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input channels: 3, Output channels: 32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes in CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 window
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to display an image.
# This function takes a tensor image, un-normalizes it, converts it to a numpy array, and then displays it.
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()  # Convert tensor to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose the dimensions to be suitable for displaying
    plt.show()

def main():
    # Download the CIFAR-10 training dataset, apply the defined transformations, and load it.
    # The dataset will be saved locally in the './data' directory.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # DataLoader is used to create mini-batches of the dataset, shuffle them, and load them in parallel using multiple workers.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Similarly, download and load the CIFAR-10 test dataset.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Define the classes/labels present in the CIFAR-10 dataset.
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Get a batch of random training images using the trainloader.
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display the images using the imshow function.
    imshow(torchvision.utils.make_grid(images))
    # Print the labels corresponding to the displayed images.
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    print('Starting evaluation')

    net.eval()  # Set the model to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total number of images

    # Ensure the model is on the CPU
    net.to('cpu')

    with torch.no_grad():  # Turn off gradients for testing, saves memory and computations
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # Get the index (class) with the maximum score for each image
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    torch.save(net.state_dict(), 'cifar10_model_cpu.pth')

    # Load with
    # model = Net()
    # model.load_state_dict(torch.load('cifar10_model_cpu.pth', map_location='cpu'))

# This ensures the script is being run as the main module and not being imported elsewhere.
if __name__ == '__main__':
    main()
