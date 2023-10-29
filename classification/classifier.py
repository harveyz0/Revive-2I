import torch
import timm
import os
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_classes(class_path=os.path.join(os.path.dirname("__file__"), "..",
                                         "data", "imagenet_classes.txt")):
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Missing image net classes file {class_path}")
    all_classes = []
    with open(class_path) as classes:
        all_classes = classes.read().splitlines()
    return all_classes


def get_args(image_dir='', output_csv='labels.csv'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=image_dir)
    parser.add_argument('--output', type=str, default=output_csv)
    return parser.parse_args()


def classify(image_dir, output_csv, classes_file):
    classes = load_classes(classes_file)

    outputs = []

    model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    for filename in os.listdir(image_dir):
        if '.png' in filename:
            img = Image.open(image_dir + os.sep + filename)
            output = model(transforms(img).unsqueeze(0))
            top1_probability, top1_index = torch.topk(output.softmax(dim=1) *
                                                      100,
                                                      k=1)
            outputs.append({
                "filename": os.path.join(image_dir, filename),
                "label": top1_index[0][0].item()
            })
            print(filename, top1_index[0][0].item(),
                  top1_probability[0][0].item(),
                  classes[top1_index[0][0].item()])

    return outputs


def main():
    args = get_args()
    classify(args.image_dir, args.output)


if __name__ == '__main__':
    main()
