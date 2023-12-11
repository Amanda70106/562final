import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import os
from transformers import CLIPProcessor, CLIPModel
import CLIP

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
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
            # input = processor(text=['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 
            #                 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 
            #                 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 
            #                 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 
            #                 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 
            #                 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 
            #                 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 
            #                 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 
            #                 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 
            #                 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 
            #                 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 
            #                 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 
            #                 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 
            #                 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 
            #                 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 
            #                 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 
            #                 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 
            #                 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 
            #                 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 
            #                 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles'], 
            #                 images=image, return_tensors="pt", padding=True)
            input = processor(text=["a dog", "a plane"], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                output = self.model(**input)
                self.attentions = output.vision_model_output.attentions
            print(type(self.attentions))
        elif model_name == "deit":
            with torch.no_grad():
                output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
    
import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    # model = torch.hub.load('facebookresearch/deit:main', 
    #     'deit_tiny_patch16_224', pretrained=True)
    # model.eval()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True)
    # if args.use_cuda:
    #     model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101", "images", "apple_pie", "112378.jpg")
    img_path = os.path.join(os.path.dirname(__file__), "plane.png")
    print(img_path)
    img = Image.open(img_path)
    img = img.resize((224, 224))
    # input_tensor = transform(img).unsqueeze(0)
    # if args.use_cuda:
    #     input_tensor = input_tensor.cuda()

    print("Doing Attention Rollout")
    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
        discard_ratio=args.discard_ratio)
    mask = attention_rollout(img, "CLIP")
    name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)



    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    # cv2.waitKey(-1)