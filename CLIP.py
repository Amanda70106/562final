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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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

folder_path = os.path.join(os.path.dirname(__file__), "Data", "food-101-imageFolder", "test")
total_num_of_pic = 0
num_of_top1_acc = [0 for i in range(101)]
num_of_top5_acc = [0 for i in range(101)]
labels = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
output_file_path = "accuracy_results.txt"  # Specify the path for the output file

with open(output_file_path, "a") as output_file:
    for food_folder in os.listdir(folder_path):
        food_folder_path = os.path.join(folder_path, food_folder)
        food_folder_index = labels.index(food_folder)
        # Check if it's a directory
        if os.path.isdir(food_folder_path):
            # Iterate through each image inside the food folder
            for image_name in os.listdir(food_folder_path):
                image_path = os.path.join(food_folder_path, image_name)
                img = Image.open(image_path)
                output = CLIP(img)
                logits_per_image = output.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                predict_index = torch.argmax(probs)
                if predict_index == food_folder_index:
                    num_of_top1_acc[food_folder_index] += 1
                if food_folder_index in torch.argsort(probs, descending=True)[:5]:
                    num_of_top5_acc[food_folder_index] += 1
                img.close()
                total_num_of_pic += 1


# Calculate top-1 and top-5 accuracy for each class
    print(num_of_top1_acc)
    print(num_of_top5_acc)
    total_top1_acc = sum(num_of_top1_acc) / total_num_of_pic if total_num_of_pic > 0 else 0
    total_top5_acc = sum(num_of_top5_acc) / total_num_of_pic if total_num_of_pic > 0 else 0

    print(f"Total Top-1 accuracy: {total_top1_acc}")
    print(f"Total Top-5 accuracy: {total_top5_acc}")

    output_file.write(f"Total Top-1 accuracy: {total_top1_acc * 100:.4f}%\n")
    output_file.write(f"Total Top-5 accuracy: {total_top5_acc * 100:.4f}%\n")




