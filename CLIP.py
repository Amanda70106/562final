from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import accuracy_score
# from PIL import Image
# import requests

def CLIP(image):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in CLIP model: {num_params}")
    input = processor(text=['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 
                            'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 
                            'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 
                            'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 
                            'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 
                            'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 
                            'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 
                            'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 
                            'French fries', 'French onion soup', 'French toast', 'Fried calamari', 
                            'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 
                            'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 
                            'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 
                            'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 
                            'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 
                            'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 
                            'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 
                            'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 
                            'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 
                            'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 
                            'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles'], 
                            images=image, return_tensors="pt", padding=True)
    # output = model(**input)
    output = 0
    return output

folder_path = os.path.join(os.path.dirname(__file__), "Data", "food-101-imageFolder", "test")
tensor_list = []
img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101-imageFolder", "test", "apple_pie", "38795.jpg")
img = Image.open(img_path)
images = ImageFolder(folder_path)
labels = []
for img, label in images:
    tensor_list.append(img)
    labels.append(label)


with torch.no_grad():
    output = CLIP(tensor_list)
logits_per_image = output.logits_per_image
# Get predicted labels
predicted_labels = torch.argmax(logits_per_image, dim=1).tolist()

# Calculate accuracy
accuracy = accuracy_score(labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

