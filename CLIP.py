import dataloader
from torchvision import transforms
food_train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
dataset = dataloader.getTrainDataset(food_train_transforms)
print(dataset)