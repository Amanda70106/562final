from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
food_train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
# train_set, val_set = dataloader.get_dataset(food_train_transforms)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Xiaolongbao-breakfast.jpg/460px-Xiaolongbao-breakfast.jpg"
image = Image.open(requests.get(url, stream=True).raw)
input = processor(text=["a image of an apple pie", "a image of dumplings"], images=image, return_tensors="pt", padding=True)
output = model(**input)
probs = output.logits_per_image.softmax(dim=1)
print(probs)
