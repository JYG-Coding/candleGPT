from __future__ import print_function


import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def download_data():
    """
    Download CIFAR-10 dataset and return training and validation datasets.
    """
    print("Downloading CIFAR-10 dataset...")
    
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    data_variance = np.var(training_data.data / 255.0)


download_data()