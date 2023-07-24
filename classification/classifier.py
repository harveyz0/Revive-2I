from urllib.request import urlopen
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import timm
import os
import argparse

def load_classes():
    # load imagenet1k class index
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    with urlopen(url) as f:
        classes = [line.strip() for line in f.readlines()]
    classes = [c.decode('utf-8') for c in classes]
    return classes
    

def main():
    classes = load_classes()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='')
    parser.add_argument('--output', type=str, default='labels.csv')
    args = parser.parse_args()

    model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    labeled_images = [] # filename, top1_index
    for filename in os.listdir(args.image_dir):
        if '.png' in filename:
            img = Image.open(args.image_dir+filename)
            output = model(transforms(img).unsqueeze(0))  
            top1_probability, top1_index = torch.topk(output.softmax(dim=1) * 100, k=1)
            labeled_images.append((filename, top1_index[0][0].item()))
            print(filename, top1_index[0][0].item(), top1_probability[0][0].item(), classes[top1_index[0][0].item()])

    with open(args.output, 'w') as f:
        f.write('filename,label\n')
        for filename, label in labeled_images:
            f.write(f'{filename},{classes[label]}\n')
    




if __name__ == '__main__':
    main()