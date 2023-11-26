import dataloader
import random
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
food_train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
train_set, val_set = dataloader.get_dataset(food_train_transforms)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
num_images = len(val_set)
random_index = random.randint(0, num_images - 1)
random_image, label = val_set[random_index] 
image = Image.open(random_image)
