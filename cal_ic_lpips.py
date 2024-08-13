import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np

lpips_fn = lpips.LPIPS(net='vgg').cuda()
preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
device='cuda'
def ic_lpips(mvtec_path,gen_path,sample_name,anomaly_name):
    print(sample_name,anomaly_name)
    tar_path = '%s/%s/%s' % (gen_path,sample_name, anomaly_name)
    ori_path='%s/%s/test/%s'%(mvtec_path,sample_name,anomaly_name)
    with torch.no_grad():
        l = len(os.listdir(ori_path)) // 3
        avg_dist = torch.zeros([l, ])
        files_list=os.listdir(tar_path)
        input_tensors1=[]
        clusters=[[] for i in range(l)]
        for k in range(l):
            input1_path = os.path.join(ori_path, '%03d.png' % k)
            input_image1 = Image.open(input1_path).convert('RGB')
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)
            input_tensors1.append(input_tensor1)
        for i in range(len(files_list)):
            min_dist = 999999999
            input2_path = os.path.join(tar_path, files_list[i])
            input_image2 = Image.open(input2_path).convert('RGB')
            input_tensor2 = preprocess(input_image2)
            input_tensor2 = input_tensor2.to(device)
            for k in range(l):
                dist = lpips_fn(input_tensors1[k], input_tensor2)
                if dist <= min_dist:
                    max_ind = k
                    min_dist = dist
            clusters[max_ind].append(input2_path)
        cluster_size=50
        for k in range(l):
            print(k)
            files_list=clusters[k]
            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    input1_path = files_list[i]
                    input2_path = files_list[j]

                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.to(device)
                    input_tensor2 = input_tensor2.to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist)
            dists = torch.tensor(dists)
            avg_dist[k] = dists.mean()
        return avg_dist[~torch.isnan(avg_dist)].mean()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mvtec_path",
        help="path ot mvtec dataset",
    )
    parser.add_argument(
        "--gen_path",
        help="path to your generated dataset",
    )
    args = parser.parse_args()
    parser = argparse.ArgumentParser()
    sample_names=[
                    'capsule',
                    'bottle',
                    'carpet',
                    'leather',
                    'pill',
                    'transistor',
                    'tile',
                    'cable',
                    'zipper',
                    'toothbrush',
                    'metal_nut',
                    'hazelnut',
                    'screw',
                    'grid',
                    'wood'
                    ]
    import csv
    for sample_name in sample_names:
        dis=0
        cnt=0
        for anomaly_name in os.listdir('%s/%s'%(args.gen_path,sample_name)):
            dis+=ic_lpips(args.mvtec_path,args.gen_path,sample_name,anomaly_name)
            cnt+=1
        with open("results.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([sample_name,str(float(dis/cnt))])