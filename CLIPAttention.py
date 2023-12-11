import torch
from PIL import Image
import cv2
from transformers import CLIPModel, CLIPProcessor
import os
import numpy as np

def CLIP_with_attention(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path)

    # Process the image and text for CLIP model input
    input = processor(
        text=[
            'Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare',
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
            'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles'
        ],
        images=image,
        return_tensors="pt",
        padding=True
        
    )

    # Get the attention weights
    outputs = model(**input)
    print(outputs.keys())
    # logits_per_image = outputs.logits_per_image
    attention_weights = outputs.vision_model_output.attentions
    print(type(attention_weights))
    # print(attention_weights)
    return attention_weights, image

import torch
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_attention_map(attentions, layer_index, head_index, sample_index, input_sequence):
    # Extract attention weights for the specified layer and head
    attention_weights = attentions[layer_index][head_index][sample_index]

    # Apply softmax
    attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=-1)

    # Ensure the dimensions of attention_weights_softmax match those of input_sequence
    attention_weights_softmax = attention_weights_softmax.unsqueeze(-1) if len(attention_weights_softmax.shape) < len(input_sequence.shape) else attention_weights_softmax

    # Create attention map
    attention_map = attention_weights_softmax * input_sequence

    # Sum across the sequence length to get the weighted average
    attention_map = attention_map.sum(dim=2).detach().numpy()

    # Normalize the attention map to values between 0 and 255 for visualization
    attention_map_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 255

    # Convert the attention map to a heatmap using seaborn
    heatmap = sns.heatmap(attention_map_normalized, cmap='viridis', annot=False, cbar=False)

    # Save the heatmap as an image
    heatmap_image_path = 'attention_map.png'
    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


# Example usage
layer_index = 0  # Adjust as needed
head_index = 0   # Adjust as needed
sample_index = 0  # Adjust as needed
input_sequence = torch.randn(1, 12, 64)  # Example input sequence (adjust dimensions as needed)

# Call the function
img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101", "images", "apple_pie", "112378.jpg")
attentions, image = CLIP_with_attention(img_path)
visualize_attention_map(attentions, layer_index, head_index, sample_index, input_sequence)




