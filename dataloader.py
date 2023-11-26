from torchvision.datasets import Food101

def getTrainDataset(transform):
    dataset = Food101(
        root="./Data", download=True, transform=transform
    )
    return dataset