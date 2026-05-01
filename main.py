
import torch
import numpy as np
import argparse
import os
import torchvision
from BATDiff.functions import create_img_scales
from BATDiff.models import BATDiffNet
from BATDiff.ModelAtrousWaveletV2 import BATDiffModelAtrousWaveletV2, MultiScaleGaussianDiffusion

from BATDiff.trainer import MultiscaleTrainer
from text2live_util.clip_extractor import ClipExtractor
from PIL import Image
import torchvision.transforms as transforms

def main():
    parser = argparse.ArgumentParser()

   
    parser.add_argument("--scope", help='choose training scope.', default='Urban25lr')
    parser.add_argument("--mode", help='choose mode: train, sample, clip_content, clip_style_gen, clip_style_trans, clip_roi, harmonization, style_transfer, roi')
    parser.add_argument("--input_image", help='content image for style transfer or harmonization.', default='seascape_composite_dragon.png')
    parser.add_argument("--start_t_harm", help='starting T at last scale for harmonization', default=5, type=int)
    parser.add_argument("--start_t_style", help='starting T at last scale for style transfer', default=15, type=int)
    parser.add_argument("--harm_mask", help='harmonization mask.', default='seascape_mask_dragon.png')
    parser.add_argument("--clip_text", help='enter CLIP text.', default='Fire in the Forest')
    parser.add_argument("--fill_factor", help='Dictates relative amount of pixels to be changed. Should be between 0 and 1.', type=float)
    parser.add_argument("--strength", help='Dictates the relative strength of CLIPs gradients. Should be between 0 and 1.',  type=float)
    parser.add_argument("--roi_n_tar", help='Defines the number of target ROIs in the new image.', default=1, type=int)

    
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='/content/drive/MyDrive/BATDiff/BATDiff/datasets/Urban25lr/')
    parser.add_argument("--image_name", help='choose image name.', default='lr.png')
    parser.add_argument("--results_folder", help='choose results folder.', default='/content/drive/MyDrive/BATDiff/BATDiff/results/Urban25lr/lr.png')

   
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=200, type=int)

  
    parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1, type=float)
    parser.add_argument("--timesteps", help='total diffusion timesteps.', default=100, type=int)

 
    parser.add_argument("--ts", help='ts.', default=4, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--train_num_steps", help='total training steps.', default=101, type=int)
    parser.add_argument("--save_and_sample_every", help='n. steps for checkpointing model.', default=100, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-3, type=float)
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.',
                        default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_milestone", help='load specific milestone.', default=0, type=int)

  
    parser.add_argument("--sample_batch_size", help='batch size during sampling.', default=1, type=int)
    parser.add_argument("--scale_mul", help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    parser.add_argument("--sample_t_list", nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)

    
    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)

    
    parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega", help='sigma=omega*max_sigma.', default=0.3, type=float)
    parser.add_argument("--loss_factor", help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

 
    parser.add_argument("--use_atrous", help='Use Atrous Wavelet model', action='store_true')
    parser.add_argument("--atrous_wavelet", help='Type of wavelet', default='b3', type=str)
    parser.add_argument("--sr_factor", help="super-resolution upscaling factor", default=8, type=int)
    parser.add_argument("--atrous_level", help='Wavelet decomposition level', default=6, type=int)
    parser.add_argument("--use_atrous_details", help='Use high-frequency details in atrous model', action='store_true')
    parser.add_argument("--details_gain", help='Gain for details in atrous model', default=0.5, type=float)

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    scale_mul = (args.scale_mul[0], args.scale_mul[1])
    sched_milestones = [val * 1000 for val in args.sched_k_milestones]
    results_folder = args.results_folder + '/' + args.scope

    save_interm = False

 
    sizes, rescale_losses, scale_factor, n_scales = create_img_scales(
    args.dataset_folder,
    args.image_name,
    scale_factor=args.scale_factor,
    image_size=None,
    create=True,
    auto_scale=None,
    atrous_level=args.atrous_level,
    sr_factor=args.sr_factor,
    detail_gain_create=1.0,
    coarse_base_gain=1.0,
    recon_blend=0.0,
    hf_boost=1.0,
    keep_same_as_input=False
)

  
    if args.use_atrous:
        print("Using Atrous Wavelet Model")
        model = BATDiffModelAtrousWaveletV2(
            dim=args.dim,
            channels=3,
            multiscale=True,
            device=device,
           
        )
    else:
        model = BATDiffNet(
            dim=args.dim,
            channels=3,
            multiscale=True,
            device=device
        )
    model.to(device)

    ms_diffusion = MultiScaleGaussianDiffusion(
        denoise_fn=model,
        save_interm=save_interm,
        results_folder=results_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        scale_mul=scale_mul,
        channels=3,
        timesteps=args.timesteps,
        train_full_t=True,
        scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l2',
        betas=None,
        device=device,
        reblurring=False,
        sample_limited_t=args.sample_limited_t,
        omega=args.omega,
        use_atrous=args.use_atrous,
        atrous_level=args.atrous_level,
        atrous_wavelet=args.atrous_wavelet,
        use_atrous_details=args.use_atrous_details,
        details_gain=args.details_gain,
    ).to(device)

    if args.sample_t_list is None:
        sample_t_list = ms_diffusion.num_timesteps_ideal[1:]
    else:
        sample_t_list = args.sample_t_list

 
    ScaleTrainer = MultiscaleTrainer(
        ms_diffusion,
        folder=args.dataset_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        ts=args.ts,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.grad_accumulate,
        ema_decay=0.995,
        fp16=False,
        save_and_sample_every=args.save_and_sample_every,
        avg_window=args.avg_window,
        sched_milestones=sched_milestones,
        results_folder=results_folder,
        device=device
    )

 
    if args.load_milestone > 0:
        ScaleTrainer.load(milestone=args.load_milestone)


    if args.mode == 'train':
        ScaleTrainer.train()
        ScaleTrainer.sample_scales(scale_mul=(1,1), custom_sample=True,
                                   image_name=args.image_name,
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=sample_t_list)
  
    elif args.mode == 'sample':
        lr_path = os.path.join(args.dataset_folder, args.image_name)
        lr_img = Image.open(lr_path).convert("RGB")
        hr_target_size = sizes[-1]   
        xref_img = lr_img.resize(hr_target_size, Image.BICUBIC)
        transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda t: (t * 2) - 1)
    ])
        lr_tensor = transform(lr_img).unsqueeze(0).to(f"cuda:{args.device_num}")
        xref_tensor = transform(xref_img).unsqueeze(0).to(f"cuda:{args.device_num}")
        ScaleTrainer.data_list[0] = (xref_tensor, xref_tensor)
        
        ms_diffusion.lr_observation = lr_tensor
        ScaleTrainer.model.lr_observation = lr_tensor
        ScaleTrainer.ema_model.lr_observation = lr_tensor
      
        ScaleTrainer.sample_scales(scale_mul=scale_mul, custom_sample=True,
                                   image_name=args.image_name,
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=sample_t_list,
                                   save_unbatched=True)


    # --- clip_content ---
    elif args.mode == 'clip_content':
        text_input = args.clip_text
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        clip_custom_t_list = sample_t_list

        guidance_sub_iters = [0] + [1]*(n_scales-1)

        assert args.strength is not None and 0 <= args.strength <= 1
        assert args.fill_factor is not None and 0 <= args.fill_factor <= 1
        strength = args.strength
        quantile = 1. - args.fill_factor
        llambda = 0.2
        stop_guidance = 3
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_sampling(clip_model=t2l_clip_extractor,
                                   text_input=text_input,
                                   strength=strength,
                                   sample_batch_size=args.sample_batch_size,
                                   custom_t_list=clip_custom_t_list,
                                   quantile=quantile,
                                   guidance_sub_iters=guidance_sub_iters,
                                   stop_guidance=stop_guidance,
                                   save_unbatched=True,
                                   scale_mul=scale_mul,
                                   llambda=llambda)

    # --- clip_style_trans / clip_style_gen ---
    elif args.mode in ['clip_style_trans','clip_style_gen']:
        text_input = args.clip_text + ' Style'
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        clip_custom_t_list = sample_t_list

        guidance_sub_iters = [0]*(n_scales-1) + [1]
        strength = 0.3
        quantile = 0.0
        llambda = 0.05
        stop_guidance = 3
        start_noise = args.mode=='clip_style_gen'
        image_name = args.image_name.rsplit(".",1)[0]+'.png'
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_sampling(clip_model=t2l_clip_extractor,
                                   text_input=text_input,
                                   strength=strength,
                                   sample_batch_size=args.sample_batch_size,
                                   custom_t_list=clip_custom_t_list,
                                   quantile=quantile,
                                   guidance_sub_iters=guidance_sub_iters,
                                   stop_guidance=stop_guidance,
                                   save_unbatched=True,
                                   scale_mul=scale_mul,
                                   llambda=llambda,
                                   start_noise=start_noise,
                                   image_name=image_name)

    # --- clip_roi ---
    elif args.mode=='clip_roi':
        text_input = args.clip_text
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        strength = 0.1
        num_clip_iters = 100
        num_denoising_steps = 3
        dataset_folder = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}/')
        image_name = args.image_name.rsplit(".", 1)[0] + '.png'
        import cv2
        image_to_select = cv2.imread(dataset_folder+image_name)
        roi = cv2.selectROI(image_to_select)
        roi_perm = [1,0,3,2]
        roi = [roi[i] for i in roi_perm]
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_roi_sampling(clip_model=t2l_clip_extractor,
                                       text_input=text_input,
                                       strength=strength,
                                       sample_batch_size=args.sample_batch_size,
                                       num_clip_iters=num_clip_iters,
                                       num_denoising_steps=num_denoising_steps,
                                       clip_roi_bb=roi,
                                       save_unbatched=True)

    # --- roi guided sampling ---
    elif args.mode=='roi':
        import cv2
        image_path = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}', args.image_name.rsplit(".", 1)[0] + '.png')
        image_to_select = cv2.imread(image_path)
        roi = cv2.selectROI(image_to_select)
        image_to_select = cv2.cvtColor(image_to_select, cv2.COLOR_BGR2RGB)
        roi_perm = [1,0,3,2]
        target_roi = [roi[i] for i in roi_perm]
        tar_y, tar_x, tar_h, tar_w = target_roi
        roi_bb_list = []
        n_targets = args.roi_n_tar
        target_h = int(image_to_select.shape[0]*scale_mul[0])
        target_w = int(image_to_select.shape[1]*scale_mul[1])
        empty_image = np.ones((target_h,target_w,3))
        target_patch_tensor = torchvision.transforms.ToTensor()(
            image_to_select[tar_y:tar_y+tar_h, tar_x:tar_x+tar_w, :])
        for i in range(n_targets):
            roi = cv2.selectROI(empty_image)
            roi_reordered = [roi[i] for i in roi_perm]
            roi_bb_list.append(roi_reordered)
            y,x,h,w = roi_reordered
            target_patch_tensor_resize = torch.nn.functional.interpolate(target_patch_tensor[None,:,:,:], size=(h,w))
            empty_image[y:y+h, x:x+w, :] = target_patch_tensor_resize[0].permute(1,2,0).numpy()
        empty_image = torchvision.transforms.ToTensor()(empty_image)
        torchvision.utils.save_image(empty_image, os.path.join(args.results_folder, args.scope, f'roi_patches.png'))
        ScaleTrainer.roi_guided_sampling(custom_t_list=sample_t_list,
                                         target_roi=target_roi,
                                         roi_bb_list=roi_bb_list,
                                         save_unbatched=True,
                                         batch_size=args.sample_batch_size,
                                         scale_mul=scale_mul)

    # --- style_transfer / harmonization ---
    elif args.mode in ['style_transfer','harmonization']:
        i2i_folder = os.path.join(args.dataset_folder, 'i2i')
        if args.mode=='style_transfer':
            start_s = n_scales-1
            start_t = args.start_t_style
            use_hist = True
        else:
            start_s = n_scales-1
            start_t = args.start_t_harm
            use_hist = False
        custom_t = [0]*(n_scales-1) + [start_t]
        hist_ref_path = f'{args.dataset_folder}scale_{start_s}/'
        ScaleTrainer.ema_model.reblurring = True
        ScaleTrainer.image2image(input_folder=i2i_folder,
                                 input_file=args.input_image,
                                 mask=args.harm_mask,
                                 hist_ref_path=hist_ref_path,
                                 batch_size=args.sample_batch_size,
                                 image_name=args.image_name,
                                 start_s=start_s,
                                 custom_t=custom_t,
                                 scale_mul=(1,1),
                                 device=device,
                                 use_hist=use_hist,
                                 save_unbatched=True,
                                 auto_scale=50000,
                                 mode=args.mode)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()
    quit()
