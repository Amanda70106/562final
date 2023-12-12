import torch
from PIL import Image
import sys
from torchvision import transforms
import numpy as np
import cv2
import os
from transformers import CLIPProcessor, CLIPModel


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.mean(axis=1)
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    

    mask = result[0, 0 , 1 :]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, image, model_name="CLIP"):
        self.attentions = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        input_tensor = transform(image).unsqueeze(0)
        if model_name == "CLIP":
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
            with torch.no_grad():
                output = self.model(**input)
                self.attentions = output.vision_model_output.attentions
            print(type(self.attentions))
        elif model_name == "deit":
            with torch.no_grad():
                output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
    



def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    # model = torch.hub.load('facebookresearch/deit:main', 
    #     'deit_tiny_patch16_224', pretrained=True)
    # model.eval()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True)

    img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101", "images", "apple_pie", "280007.jpg")
    # img_path = os.path.join(os.path.dirname(__file__), "plane.png")
    print(img_path)
    img = Image.open(img_path)
    img = img.resize((224, 224))

    print("Doing Attention Rollout")
    attention_rollout = VITAttentionRollout(model)
    mask = attention_rollout(img, "CLIP")
    name = "attention_rollout_{:.3f}_{}.png".format(0.9, "mean")



    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    # cv2.waitKey(-1)