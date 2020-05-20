from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import skimage
from skimage import transform
import time
'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, depth, nmaps, intrinsics):
        for t in self.transforms:
            images, depth, nmaps, intrinsics = t(images, depth, nmaps, intrinsics)
        return images, depth, nmaps, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, depth, nmaps, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, depth, nmaps, intrinsics

class DownSample(object):
    def __init__(self,scale):
        self.scale = scale
    def __call__(self, images, depth, nmaps, intrinsics):
        dn_maps = []
        for n in nmaps:
            dn_maps.append(torch.nn.functional.interpolate(n.unsqueeze(0),scale_factor = self.scale).squeeze(0))
        return images, depth, dn_maps, intrinsics

class NormalCheck(object):
    
    def __call__(self, images, depth, nmaps, intrinsics):
        
        temp = nmaps[0][1].clone()
        nmaps[0][1] = nmaps[0][2].clone()*-1
        nmaps[0][2] = temp
        return images, depth, nmaps, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, depth, nmaps, intrinsics):
        tensors = []
        ntensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        
        for nm in nmaps:
            nm = np.transpose(nm, (2, 0, 1))
            ntensors.append(torch.from_numpy(nm).float())

        return tensors, depth, ntensors, intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, depths, nmaps, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h = 240
        out_w = 320
        in_h, in_w, _ = images[0].shape
        x_scaling = np.random.uniform(out_w/in_w, 1)
        y_scaling = np.random.uniform(out_h/in_h, 1)
        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling


        scaled_images = [skimage.transform.rescale(im, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for im in images]
        
        scaled_nmaps = [skimage.transform.rescale(nm, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for nm in nmaps]
        
        scaled_depths = [skimage.transform.rescale(dp, scale=(y_scaling, x_scaling),multichannel = False, preserve_range = True).astype(np.float32) for dp in depths]
        
        np.random.seed(int(time.time()))
        offset_y = np.random.randint(scaled_h - out_h + 1)
        offset_x = np.random.randint(scaled_w - out_w + 1)
        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_nmaps = [nm[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for nm in scaled_nmaps]
        cropped_depths = [dp[offset_y:offset_y + out_h, offset_x:offset_x + out_w] for dp in scaled_depths] 

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        
        return cropped_images, cropped_depths, cropped_nmaps, output_intrinsics


class Scale(object):
    def __call__(self, images, depths, nmaps, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h = 240
        out_w = 320
        in_h, in_w, _ = images[0].shape
        x_scaling = out_w/in_w
        y_scaling = out_h/in_h
        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling


        scaled_images = [skimage.transform.rescale(im, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for im in images]
        
        scaled_nmaps = [skimage.transform.rescale(nm, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for nm in nmaps]
        
        scaled_depths = [skimage.transform.rescale(dp, scale=(y_scaling, x_scaling),multichannel = False, preserve_range = True).astype(np.float32) for dp in depths]
        
        offset_y = 0 
        offset_x = 0 
        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_nmaps = [nm[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for nm in scaled_nmaps]
        cropped_depths = [dp[offset_y:offset_y + out_h, offset_x:offset_x + out_w] for dp in scaled_depths] 

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        
        return cropped_images, cropped_depths, cropped_nmaps, output_intrinsics

class RandomCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""
    def __init__(self,scale = 1, h = 240, w = 320):
        self.scale = scale
        self.h = h
        self.w = w
    def __call__(self, images, depths, nmaps, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h = self.h
        out_w = self.w
        in_h, in_w, _ = images[0].shape
        x_scaling = self.scale
        y_scaling = self.scale
        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling


        if x_scaling != 1 or y_scaling != 1:
            scaled_images = [skimage.transform.rescale(im, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for im in images]

            scaled_nmaps = [skimage.transform.rescale(nm, scale=(y_scaling, x_scaling), multichannel = True, preserve_range = True).astype(np.float32) for nm in nmaps]
            
            scaled_depths = [skimage.transform.rescale(dp, scale=(y_scaling, x_scaling),multichannel = False, preserve_range = True).astype(np.float32) for dp in depths]
        else:
            scaled_images = images
            scaled_nmaps = nmaps
            scaled_depths = depths

        


        np.random.seed(int(time.time()))
        offset_y = np.random.randint(scaled_h - out_h + 1)
        offset_x = np.random.randint(scaled_w - out_w + 1)
        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_nmaps = [nm[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for nm in scaled_nmaps]
        cropped_depths = [dp[offset_y:offset_y + out_h, offset_x:offset_x + out_w] for dp in scaled_depths] 


        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        
        return cropped_images, cropped_depths, cropped_nmaps, output_intrinsics
