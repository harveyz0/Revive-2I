"""
evaluates the output images
Metrics: FID, KID, Resnet50 top-k accuracy
"""

import os
import numpy as np
import requests
import torch
import timm
from urllib.request import urlopen
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from fid_score import calculate_fid_given_paths
from kid_score import calculate_kid_given_paths

import argparse


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

def calculate_kid(generated, real, cuda=True, batch_size=50, dims=2048):
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
    with open('dog_list.txt', 'r') as f:
        for line in f:
            dog_list.append(line.split('\t')[2].strip())

    for dog in dog_list:
        if ',' in dog:
            new_name = dog.split(',')[0].strip()
            dog_list[dog_list.index(dog)] = new_name
    return dog_list

def call_resnet(image, token):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
    bearer = "Bearer " + token
    headers = {"Authorization": bearer}
    
    with open(image, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    response = response.json()

    return response

def call_api(generated_dir, token):
    responses = []
    for filename in os.listdir(generated_dir):
        if filename.endswith(".png"):
            responses.append(call_resnet(os.path.join(generated_dir, filename), token))
    return responses


def call_accuracy(responses, generated_dir):
    breeds = get_dog_breeds()
    top1 = 0
    top5 = 0
    for response in responses:
        if 'error' in response:
            top1 += 1
            top5 += 1
            continue 
        if response[0]['label'] in breeds:
            top1 += 1
        rep5 = response[:5]
        labels = [x['label'] for x in rep5]
        if any(label in breeds for label in labels):
            top5 += 1

    return top1/len(os.listdir(generated_dir)), top5/len(os.listdir(generated_dir))


def top1_accuracy(generated_dir, token):
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

def top5_accuracy(generated_dir, token):
    top5 = 0
    breeds = get_dog_breeds()
    for filename in os.listdir(generated_dir):
        if filename.endswith(".png"):
            response = call_resnet(os.path.join(generated_dir, filename))
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
        '--api-key',
        type=str,
        default=None,
        help='api key for resnet50'
    )

    parser.add_argument(
        '--class-csv',
        type=str,
        default=None,
        help='path to csv file with class labels'
    )

    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='path to save the evaluation results'
    )

    opt = parser.parse_args()

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] != '-1' else torch.device('cpu')

    fid = calculate_fid(opt.generated_dir, opt.target_dir, cuda=(opt.gpu_ids[0] != '-1'))
    print('FID: {}'.format(fid))
    kid = calculate_kid(opt.generated_dir, opt.target_dir, cuda=(opt.gpu_ids[0] != '-1'))
    print('KID: {}'.format(kid))
    
    if opt.api_key:
        responses = call_api(opt.generated_dir, opt.api_key)
        top1, top5 = call_accuracy(responses, opt.generated_dir)
        print('top1: {}'.format(top1))
        print('top5: {}'.format(top5))
    elif opt.class_csv:
        #csv format: filename,label 
        with open(opt.class_csv, 'r') as f:
            lines = f.readlines()
            label_gold_pairs = []
            for line in lines:
                label = line.split(',')[1].strip()
                filename = line.split(',')[0].strip()
                gold = ''
                if 'boston' in filename:
                    gold = 'Boston bull'
                elif 'boxer' in filename:
                    gold = 'boxer'
                elif 'chi' in filename:
                    gold = 'Chihuahua'
                elif 'dane' in filename:
                    gold = 'Great Dane'
                elif 'pek' in filename:
                    gold = 'Pekinese'
                elif 'rot' in filename:
                    gold = 'Rottweiler'
                else:
                    continue
                pair = (label, gold)
                label_gold_pairs.append(pair)
    
        dog_list = get_dog_breeds()
        top1_dog = 0 # First score: how many are dog labels (use dog_list)
        top1_gold = 0 # Second score: how many are the gold label

        for pair in label_gold_pairs:
            if pair[0] in dog_list:
                top1_dog += 1
            if pair[0] == pair[1]:
                top1_gold += 1

        top1_dog /= len(label_gold_pairs)
        top1_gold /= len(label_gold_pairs)
        print('top1_dog: {}'.format(top1_dog))
        print('top1_gold: {}'.format(top1_gold))

        if opt.out_dir:
            with open(os.path.join(opt.out_dir, 'scores.txt'), 'w') as f:
                f.write('FID: {}\n'.format(fid))
                f.write('KID: {}\n'.format(kid))
                f.write('top1_dog: {}\n'.format(top1_dog))
                f.write('top1_gold: {}\n'.format(top1_gold))
        else:
            print('FID: {}\n'.format(fid))
            print('KID: {}\n'.format(kid))
            print('top1_dog: {}\n'.format(top1_dog))
            print('top1_gold: {}\n'.format(top1_gold))








if __name__ == '__main__':
    main()

