import torch
import timm
import os
import argparse
from urllib.request import urlopen
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_classes():
    # load imagenet1k class index
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    with urlopen(url) as f:
        classes = [line.strip() for line in f.readlines()]
    classes = [c.decode('utf-8') for c in classes]
    return classes


def get_args(image_dir='', output_csv='labels.csv'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=image_dir)
    parser.add_argument('--output', type=str, default=output_csv)
    return parser.parse_args()


def classify(image_dir, output_csv):
    classes = load_classes()

    model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    labeled_images = []  # filename, top1_index
    for filename in os.listdir(image_dir):
        if '.png' in filename:
            img = Image.open(image_dir + os.sep + filename)
            output = model(transforms(img).unsqueeze(0))
            top1_probability, top1_index = torch.topk(output.softmax(dim=1) *
                                                      100,
                                                      k=1)
            labeled_images.append((filename, top1_index[0][0].item()))
            print(filename, top1_index[0][0].item(),
                  top1_probability[0][0].item(),
                  classes[top1_index[0][0].item()])

    with open(output_csv, 'w') as f:
        f.write('filename,label\n')
        for filename, label in labeled_images:
            f.write(f'{filename},{classes[label]}\n')


def main():
    args = get_args()
    classify(args.image_dir, args.output)


if __name__ == '__main__':
    main()
