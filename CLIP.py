from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from PIL import Image
# from PIL import Image
# import requests

def CLIP(image):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
    output = model(**input)
    return output


