# Import necessary libraries
import torch  # PyTorch library for tensor computations and deep learning
import torchvision  # Extension of PyTorch for computer vision tasks
import torchvision.transforms as transforms  # Tools for image transformations
import matplotlib.pyplot as plt  # Library for plotting and visualization
import numpy as np  # Library for numerical computations

# Define a series of transformations to apply to the images:
# 1. Convert images to PyTorch tensors.
# 2. Normalize the images so that their pixel values are between -1 and 1.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

# This ensures the script is being run as the main module and not being imported elsewhere.
if __name__ == '__main__':
    main()
