import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from transformers import CLIPProcessor, CLIPModel
import cv2

def rollout(attentions, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.min(axis=1)[0]

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

    mask = result[0, 0 , 1 :]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

class AttenMap:
    def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9): 
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio)
# only for CLIP
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
if __name__ == '__main__':
    model = model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101", "images", "apple_pie", "134.jpg")
    img = Image.open(os.path.__file__)
    img = img.resize((224, 224))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(img).unsqueeze(0)
    attention_rollout = AttenMap(model)
    mask = attention_rollout(input_tensor)
    name = "attention_rollout_{:.3f}_{}.png".format(0.9, "mean")
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    # cv2.waitKey(-1)