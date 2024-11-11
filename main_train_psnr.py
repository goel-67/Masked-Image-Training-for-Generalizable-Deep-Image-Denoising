import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import lpips
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

'''
# --------------------------------------------
# Training code for MSRResNet with added noise types  # <-- Updated comment to reflect changes
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

# Disable CuDNN for reproducibility or compatibility reasons
torch.backends.cudnn.enabled = False

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    """
    Converts a numpy image to a PyTorch tensor.

    Args:
        image (numpy.ndarray): Input image.
        imtype (type): Desired data type.
        cent (float): Centroid for normalization.
        factor (float): Scaling factor.

    Returns:
        torch.Tensor: Converted tensor.
    """
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

# Added function `compute_current_ratio` to calculate the current mask ratio
def compute_current_ratio(current_step, total_steps, initial_ratio, final_ratio):
    """
    Compute the current mask ratio using linear decay.

    Args:
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        initial_ratio (float): Initial mask ratio.
        final_ratio (float): Final mask ratio.

    Returns:
        float: Current mask ratio.
    """
    ratio = initial_ratio - (initial_ratio - final_ratio) * (current_step / total_steps)
    return max(ratio, final_ratio)  # Ensure it doesn't go below the final ratio

def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False, action='store_true', help='Use distributed training')  # <-- Added help description for distributed training

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist

    # Ensure noise types are defined in options  # <-- Added to ensure noise types exist
    if 'datasets' in opt and 'train' in opt['datasets']:
        opt['datasets']['train']['noise_types'] = opt['datasets']['train'].get('noise_types', ['impulse', 'quantization', 'thermal', 'poisson'])

    writer = SummaryWriter('./runs/' + opt['task'])

    # ----------------------------------------
    # Distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs([path for key, path in opt['path'].items() if 'pretrained' not in key])

    # ----------------------------------------
    # Update opt
    # ----------------------------------------
    # Find last checkpoints
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # Uncomment to start from step 0
    # current_step = 0  # <-- Comment added to indicate option to reset step

    border = opt['scale']

    # ----------------------------------------
    # Save opt to a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # Return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # Configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # Seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) Create dataset
    # 2) Create dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                # Adjust batch size and number of workers based on number of GPUs
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'])
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    # -------------------------------
    # Define initial and final mask ratios  # <-- Added mask ratio initialization section
    # -------------------------------
    initial_mask_ratio1 = 0.8
    final_mask_ratio1 = 0.2
    initial_mask_ratio2 = 0.8
    final_mask_ratio2 = 0.2

    # -------------------------------
    # Set fixed hyperparameters  # <-- Added fixed hyperparameters section
    # -------------------------------
    fixed_learning_rate = 1e-3  # Set to desired fixed learning rate
    fixed_dropout_rate = 0.3     # Set to desired fixed dropout rate

    logger.info(f'Training with fixed learning rate: {fixed_learning_rate} and fixed dropout rate: {fixed_dropout_rate}')

    # Update the options with fixed hyperparameters
    opt['train']['G_optimizer_lr'] = fixed_learning_rate
    opt['netG']['dropout_rate'] = fixed_dropout_rate

    # Initialize model with updated options
    model = define_Model(opt)
    model.init_train()

    # Reset current_step for training  # <-- Added comment to indicate current step reset
    current_step = 0

    # Get maximum epochs from options  # <-- Added to retrieve max epochs from options
    max_epochs = opt.get('train', {}).get('max_epochs', 1100)

    # Calculate total steps  # <-- Added for total step calculation
    total_steps = max_epochs * train_size

    logger.info(f'Total Training Steps for Progressive Masking: {total_steps}')

    # ==================================================================
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    best_PSNRY = 0
    best_step = 0    
    # ==================================================================

    for epoch in range(max_epochs):  # Use max_epochs from options  # <-- Updated loop to use max_epochs
        if opt['dist']:
            train_loader.sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):
            current_step += 1

            # Compute current mask ratios  # <-- Added to compute mask ratios
            mask_ratio1_current = compute_current_ratio(
                current_step, total_steps, initial_mask_ratio1, final_mask_ratio1
            )
            mask_ratio2_current = compute_current_ratio(
                current_step, total_steps, initial_mask_ratio2, final_mask_ratio2
            )

            # Set the current mask ratios in the model  # <-- Added to set mask ratios
            model.mask_ratio1 = mask_ratio1_current
            model.mask_ratio2 = mask_ratio2_current

            # Optionally, log the current mask ratios  # <-- Added mask ratio logging
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logger.info(
                    f'Current Step: {current_step}, '
                    f'Mask Ratio1: {mask_ratio1_current:.4f}, '
                    f'Mask Ratio2: {mask_ratio2_current:.4f}'
                )
                # Log to TensorBoard  # <-- Added TensorBoard logging for mask ratios
                writer.add_scalar('Mask Ratio1', mask_ratio1_current, global_step=current_step)
                writer.add_scalar('Mask Ratio2', mask_ratio2_current, global_step=current_step)

            # -------------------------------
            # 1) Update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) Feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) Optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) Training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                    # Log to TensorBoard  # <-- Added detailed TensorBoard logging
                    writer.add_scalar('loss/' + k, v, global_step=current_step)

                logger.info(message)

            # -------------------------------
            # 5) Save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info(f'Saving the model at step {current_step}.')
                model.save(current_step)

            # -------------------------------
            # 6) Testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrY = 0.0
                avg_ssimY = 0.0
                avg_lpips = 0.0
                idx = 0
                save_list = []

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # Save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # Calculate PSNR, SSIM, LPIPS
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    current_lpips = loss_fn_alex(im2tensor(E_img).cuda(), im2tensor(H_img).cuda()).item()

                    # Convert to Y channel for PSNR_Y and SSIM_Y
                    if E_img.ndim == 3:  # RGB image
                        output_y = util.bgr2ycbcr(E_img.astype(np.float32) / 255.) * 255.
                        img_gt_y = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
                        psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                        ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                    else:
                        # If grayscale, Y channel is the image itself
                        psnr_y = current_psnr
                        ssim_y = current_ssim

                    # Aggregate results
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_lpips += current_lpips
                    avg_psnrY += psnr_y
                    avg_ssimY += ssim_y

                    logger.info(
                        f'{idx:->4d}--> {image_name_ext:>20s} | PSNR: {current_psnr:<4.2f} dB; SSIM: {current_ssim:<5.4f}; '
                        f'PSNR_Y: {psnr_y:<4.2f} dB; SSIM_Y: {ssim_y:<5.4f}; LPIPS: {current_lpips:<5.4f}'
                    )

                    if img_name in opt['train']['save_image']:
                        logger.info(f'Saving test image: {img_name}')
                        save_list.append(util.uint2tensor3(E_img)[:, :512, :512])

                # Calculate averages
                avg_psnr /= idx
                avg_ssim /= idx
                avg_psnrY /= idx
                avg_ssimY /= idx
                avg_lpips /= idx

                if len(save_list) > 0 and current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                    save_images = make_grid(save_list, nrow=len(save_list))
                    writer.add_image("test", save_images, global_step=current_step)

                if avg_psnrY >= best_PSNRY:
                    best_step = current_step
                    best_PSNRY = avg_psnrY

                # Log average metrics
                logger.info(
                    f'<epoch:{epoch:3d}, iter:{current_step:8,d}, Average: PSNR: {avg_psnr:.2f} dB; SSIM: {avg_ssim:.4f}; '
                    f'PSNR_Y: {avg_psnrY:.2f} dB; SSIM_Y: {avg_ssimY:.4f}; LPIPS: {avg_lpips:.4f}>'
                )
                logger.info(f'--- Best PSNRY --->   iter:{best_step:8,d}, Average: PSNR_Y: {best_PSNRY:.2f} dB\n')

                # Log to TensorBoard
                writer.add_scalar('PSNRY', avg_psnrY, global_step=current_step)
                writer.add_scalar('SSIMY', avg_ssimY, global_step=current_step)
                writer.add_scalar('PSNR', avg_psnr, global_step=current_step)
                writer.add_scalar('SSIM', avg_ssim, global_step=current_step)
                writer.add_scalar('LPIPS', avg_lpips, global_step=current_step)

    # ----------------------------------------
    # Save the final model after training
    # ----------------------------------------
    if opt['rank'] == 0:
        logger.info('Training completed. Saving the final model.')
        final_model_path = os.path.join(opt['path']['models'], 'final_model.pth')
        torch.save(model.netG.state_dict(), final_model_path)
        logger.info(f'Final model saved to {final_model_path}')

        final_optimizer_path = os.path.join(opt['path']['models'], 'final_optimizerG.pth')
        torch.save(model.G_optimizer.state_dict(), final_optimizer_path)
        logger.info(f'Final optimizer saved to {final_optimizer_path}')

    writer.close()

if __name__ == '__main__':
    main()
