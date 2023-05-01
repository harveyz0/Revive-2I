"""
evaluates the output images
Metrics: FID, KID, Resnet50 top-k accuracy
"""

import os
import numpy as np
import requests
import torch
from fid_score import calculate_fid_given_paths
from kid_score import calculate_kid_given_paths

import argparse


# def calculate_fid_given_paths(paths, batch_size, cuda, dims, bootstrap=True, n_bootstraps=10, model_type='inception'):
def calculate_fid(generated, real, cuda=True, batch_size=25, dims=2048):
    count = len(os.listdir(real))
    incrementer = 1
    while count > incrementer:
        if count % incrementer == 0:
            batch_size = incrementer
        incrementer += 1
    print('batch size: {}'.format(batch_size))

    fid = calculate_fid_given_paths([generated, real], batch_size, cuda, dims)
    return fid

# calculate_kid_given_paths(paths, batch_size, cuda, dims, model_type='inception'):
def calculate_kid(generated, real, device, cuda=True, batch_size=50, dims=2048):
    count = len(os.listdir(real))
    incrementer = 1
    while count > incrementer:
        if count % incrementer == 0:
            batch_size = incrementer
        incrementer += 1
    print('batch size: {}'.format(batch_size))

    kid = calculate_kid_given_paths([generated, real], batch_size, cuda, dims)
    return kid

def get_dog_breeds():
    # from dog_list.txt, get all the dog breed labels. they are listed as
    # 0	151	Chihuahua
    dog_list = []
    with open('eval/dog_list.txt', 'r') as f:
        for line in f:
            dog_list.append(line.split('\t')[2].strip())
    print(dog_list)
    return dog_list

def call_resnet(image):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
    

    with open(image, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    response = response.json()
    # get top 1 label from response and return it 
    return response


def top1_accuracy(generated_dir):
    top1 = 0
    breeds = get_dog_breeds()
    for filename in os.listdir(generated_dir):
        if filename.endswith(".png"):
            response = call_resnet(os.path.join(generated_dir, filename))
            if response[0]['label'] in breeds:
                top1 += 1
    print('top1')
    print(top1)
    print(len(os.listdir(generated_dir)))
    print(top1/len(os.listdir(generated_dir)))

def top5_accuracy(generated_dir):
    top5 = 0
    breeds = get_dog_breeds()
    for filename in os.listdir(generated_dir):
        if filename.endswith(".png"):
            response = call_resnet(os.path.join(generated_dir, filename))
            # print(response)
            # response = response[:5]
            # print(response)
            labels = [x['label'] for x in response]
            if any(label in breeds for label in labels):
                top5 += 1
    print('top5')
    print(top5)
    print(len(os.listdir(generated_dir)))
    print(top5/len(os.listdir(generated_dir)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--generated-dir', 
        type=str, 
        default='outputs/img2img-samples/prompt',
        help='path to the generated images'
    )

    parser.add_argument(
        '--target-dir', 
        type=str, 
        default='datasets/skull2animal/testB',
        help='path to the target images'
    )

    parser.add_argument(
        '--gpu-ids',
        type=str,
        default='0',
        help='gpu ids: e.g. 0  0,1,2, 0,2.; -1 for CPU'
    )

    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs/eval',
        help='path to save the evaluation results'
    )

    opt = parser.parse_args()

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] != '-1' else torch.device('cpu')

    fid = calculate_fid(opt.generated_dir, opt.target_dir)
    print('FID: {}'.format(fid))
    kid = calculate_kid(opt.generated_dir, opt.target_dir, device)
    print('KID: {}'.format(kid))
    print(test_resnet('C:\\Users\\amart50\\Documents\\stable-i2i\\outputs\\img2img-samples\\a_realistic_photo_of_a_dog_head\\seed_24469_00077.png'))
    top1_accuracy(opt.generated_dir)
    top5_accuracy(opt.generated_dir)





if __name__ == '__main__':
    main()

