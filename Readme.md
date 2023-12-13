## Swin transformer implementation
Yudong Li

-- Borrowed the implementation of the SwinTransformer model from the official github https://github.com/microsoft/Swin-Transformer

-- Made the model compatible with Food-101 datset

-- Modified the training code to work on single GPU machine

-- Organized the Food-101 datset

-- Train the model by going into ./swin directory and run main.py


## Deit implementation
Zhiwei Zhong

-- Borrowed the implementation of the DeiT model from the official github https://github.com/facebookresearch/deit/blob/main/README_deit.md

-- Made the model compatible with Food-101 datset

-- Modified the training code to work on single GPU machine

-- Train the model by going into ./deit directory and run python deit_train.py --model deit_small_patch16_224 --batch-size 16 --data-path ../Data/food-101-imageFolder --output_dir ./output

## CLIP implementation
Pei-Hsuan Lin

-- Used the official pretrained model from the official github https://github.com/openai/CLIP.git

-- Made the model compatible with Food-101 datset

-- Implemented the dataloader

-- Borrowed and adapted the attention visualization code from https://github.com/jacobgil/vit-explain/blob/main/vit_explain.py


