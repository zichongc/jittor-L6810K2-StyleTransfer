import jittor as jt
import jittor.transform as transform
import json
import numpy as np
from PIL import Image
from typing import List, Union


def jtvar_to_pil(images: jt.Var) -> List[Image.Image]:
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pil_to_jtvar(images: Union[Image.Image, List[Image.Image]]):
    images = transform.ToTensor()(images)
    return images


def save_image(img, path, nrow=10, padding=2, normalize=False, format='RGB'):
    N, C, H, W = img.shape
    if N % nrow != 0:
        print("N % nrow != 0")
        return
    ncol = N // nrow
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,H,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    if normalize:
        min_ = img.min()
        max_ = img.max()
        img = (img-min_)/(max_-min_)
    img *= 255
    img = img.transpose((1,2,0))
    if C == 3:
        if format == 'RGB':
            img = img[:,:,:] 
        elif format == 'BGR':
            img = img[:,:,::-1]
    elif C == 1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)


def load_test_prompts(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    prompts = list(data.values())
    print(prompts)
    return prompts
