import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_mask

class DatasetMaskedDenoising(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN with added noise types  # <-- Added comment to indicate an enhancement
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetMaskedDenoising, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 1
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize * self.sf

        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        print(f'len(self.paths_H): {len(self.paths_H)}')

        assert self.paths_H, 'Error: H path is empty.'
        
        self.if_mask = True if opt['if_mask'] else False
        self.noise_types = opt.get('noise_types', ['impulse', 'quantization', 'thermal', 'poisson'])  # <-- Added attribute for noise types

    def apply_noise(self, image):  # <-- Added new method for applying noise
        noise_type = random.choice(self.noise_types)
        if noise_type == 'impulse':
            return util.add_impulse_noise(image)
        elif noise_type == 'quantization':
            return util.add_quantization_noise(image)
        elif noise_type == 'thermal':
            return util.add_thermal_noise(image)
        elif noise_type == 'poisson':
            return util.add_poisson_noise(image)
        else:
            return image  # Default to no noise if not recognized

    def __getitem__(self, index):
        L_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
            img_H = self.apply_noise(img_H)  # <-- Added line to apply noise during training

            img_L, img_H = utils_mask.input_mask_with_noise(
                img_H, 
                sf=self.sf, 
                lq_patchsize=self.lq_patchsize, 
                noise_level=self.opt['noise_level'], 
                if_mask=self.if_mask, 
                mask1=self.opt['mask1'], 
                mask2=self.opt['mask2']
            )
        else:
            img_H = util.uint2single(img_H)
            img_H = self.apply_noise(img_H)  # <-- Added line to apply noise during non-training phase
            img_L, img_H = utils_mask.input_mask_with_noise(img_H, self.sf, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        
        if L_path is None:
            L_path = H_path
                        
        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
