from torchvision import datasets
from torchvision import transforms
# print(dir(datasets))
def get_dataset(transform):
    train_set = datasets.Food101(
        root="./Data", download=True, transform=transform
        , split="train"
    )
    val_set = datasets.Food101(
        root="./Data", download=True, transform=transform
        , split="test"
    )
    return train_set, val_set

