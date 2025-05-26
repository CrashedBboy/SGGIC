import math
import io
import shutil
import os
import sys
from os import path
import statistics

import numpy as np
import clip
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getDataset(image_dir):
    dataset = []
    image_files = [f for f in os.listdir(image_dir) if ".png" in f and path.isfile(os.path.join(image_dir, f))]
    for f in image_files:
        img_id = f.split(".png")[0]
        img_path = path.join(image_dir, f)
        semantics = {
            "object": [],
            "detail": [],
            "summary": ""
        }
        semantics_path = path.join(image_dir, f"{img_id}.semantic.txt")
        with open(semantics_path, "r") as semantic_f:
            lines = semantic_f.read().split('\n')
            semantics["object"] += lines[0:3]
            semantics["detail"] += lines[3:6]
            semantics["summary"] = str(lines[6])
        data = {
            'image_path': img_path,
            'image_name': img_id,
            'semantics': semantics
        }
        dataset.append(data)
    return dataset