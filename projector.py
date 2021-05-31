'''Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/projector.py'''

import paddle
from paddle import optimizer as optim
from paddle.nn import functional as F
from paddle.vision import transforms

import paddle_lpips as lpips
from utils import arg_type, func_args, get_generator, get_pSp

import PIL
from PIL import Image
from tqdm import tqdm
from typing import List


def get_lr(t, ts, initial_lr, final_lr):
    alpha = pow(final_lr/initial_lr, 1/ts)**(t*ts)

    return initial_lr * alpha


def make_image(tensor):
    return (
        ((tensor.detach() + 1) / 2 * 255)
        .clip(min=0, max=255)
        .transpose((0, 2, 3, 1))
        .numpy()
        .astype('uint8')
    )


def project(
    imgs: List[PIL.Image.Image],
    masks: List[PIL.Image.Image] = None,
    generator = None,
    pSp = None,
    ckpt: arg_type(str, help="path to the model checkpoint") = None,
    model_type: arg_type(str, help="inner model type. `ffhq-config-f` for default genrator and `ffhq-inversion` for pSp") = None,
    size: arg_type(int, help="original output image resolution") = 1024,
    style_dim: arg_type(int, help="dimensions of style z") = 512,
    n_mlp: arg_type(int, help="the number of multi-layer perception layers for style z") = 8,
    channel_multiplier: arg_type(int, help="channel product, affect model size and the quality of generated pictures") = 2,
    start_lr: arg_type(float, help="learning rate at the begin of training") = 0.1,
    final_lr: arg_type(float, help="learning rate at the end of training") = 0.025,
    latent_level: arg_type(List[int], help="indices of latent code for training") = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
    step: arg_type(int, help="optimize iterations") = 100,
    mse_weight: arg_type(float, help="weight of the mse loss") = 1,
    no_encoder: arg_type(
       'project:no_encoder', action="store_true", 
       help="disable to use pixel2style2pixel model to pre-encode the images"
    ) = False,
):
    n_mean_latent = 4096

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Transpose(),
            transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
        ]
    )

    _imgs = []
    _masks = []
    if masks is None:
        masks = [Image.new(mode='L', size=img.size, color=255) for img in imgs]
    for img, mask in zip(imgs, masks):
        assert isinstance(img, PIL.Image.Image) and isinstance(mask, PIL.Image.Image)
        img = paddle.to_tensor(transform(img))
        mask = (paddle.to_tensor(transform(mask.convert('RGB')))[:1] + 1) / 2
        _imgs.append(img)
        _masks.append(mask)

    imgs = paddle.stack(_imgs, 0)
    masks = paddle.stack(_masks, 0)

    percept = lpips.LPIPS(net='vgg')
    percept.train() # on PaddlePaddle, lpips's default eval mode means no gradients.

    if generator is not None:
        no_encoder = True
    if pSp is None:
        no_encoder = False
    if no_encoder:
        generator = generator if generator is not None else get_generator(
            weight_path=None if ckpt is None else ckpt,
            model_type='ffhq-config-f' if model_type is None else model_type,
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier
        )
        # generator.eval() # on PaddlePaddle, model.eval() means no gradients.

        with paddle.no_grad():
            noise_sample = paddle.randn((n_mean_latent, style_dim))
            latent_out = generator.style(noise_sample)

            latent_mean = latent_out.mean(0)

        latent_in = latent_mean.detach().clone().unsqueeze(0).tile((imgs.shape[0], 1))
        latent_in = latent_in.unsqueeze(1).tile((1, generator.n_latent, 1)).detach()

    else:
        pSp = pSp if pSp is not None else get_pSp(
            weight_path=None if ckpt is None else ckpt,
            model_type='ffhq-inversion' if model_type is None else model_type,
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier
        )
        # pSp.eval() # on PaddlePaddle, model.eval() means no gradients.
        generator = pSp.decoder
        
        with paddle.no_grad():
            _, latent_in = pSp(imgs, randomize_noise=False, return_latents=True)
        latent_in = latent_in.detach().clone()
    
    var_levels = list(latent_level)
    const_levels = [i for i in range(generator.n_latent) if i not in var_levels]
    assert len(var_levels) > 0
    if len(const_levels) > 0:
        latent_fix = latent_in.index_select(paddle.to_tensor(const_levels), 1).detach().clone()
        latent_in = latent_in.index_select(paddle.to_tensor(var_levels), 1).detach().clone()

    latent_in.stop_gradient = False

    optimizer = optim.Adam(parameters=[latent_in], learning_rate=start_lr)

    frames = []

    pbar = tqdm(range(step))

    for i in pbar:
        t = i / step
        lr = get_lr(t, step, start_lr, final_lr)
        optimizer.set_lr(lr)

        if len(const_levels) > 0:
            latent_dict = {}
            for idx, idx2 in enumerate(var_levels):
                latent_dict[idx2] = latent_in[:,idx:idx+1]
            for idx, idx2 in enumerate(const_levels):
                latent_dict[idx2] = (latent_fix[:,idx:idx+1]).detach()
            latent_list = []
            for idx in range(generator.n_latent):
                latent_list.append(latent_dict[idx])
            latent_n = paddle.concat(latent_list, 1)
        else:
            latent_n = latent_in

        img_gen, _ = generator([latent_n], input_is_latent=True, randomize_noise=False)
        frames.append(make_image(img_gen))

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                (batch, channel, height // factor, factor, width // factor, factor)
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen*masks, (imgs*masks).detach()).sum()
        mse_loss = F.mse_loss(img_gen*masks, (imgs*masks).detach())
        loss = p_loss + mse_weight * mse_loss

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"perceptual: {p_loss.numpy()[0]:.4f}; "
                f"mse: {mse_loss.numpy()[0]:.4f}; lr: {lr:.4f}"
            )
        )

    img_gen, _ = generator([latent_n], input_is_latent=True, randomize_noise=False)
    frames.append(make_image(img_gen))

    imgs_seq = [[] for _ in range(img_gen.shape[0])]
    for i in range(img_gen.shape[0]):
        for frame in frames:
            imgs_seq[i].append(frame[i])
    
    return imgs_seq, latent_n


if __name__ == "__main__":
    import argparse
    import os

    from utils import save_video
    from crop import align_face

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser, arg_names = func_args(parser, project)
    parser.add_argument(
        "--no_crop", action="store_true", help="disable to crop input images first"
    )
    parser.add_argument(
        "--save_mp4", action="store_true", help="saving training progress images as mp4 videos"
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    imgs = []
    masks = []

    for imgfile in args.files:
        if args.no_crop:
            img = Image.open(imgfile)
            imgs.append(img)
            maskfile = '.'.join(imgfile.split('.')[:-1]) + '.mask.' + imgfile.split('.')[-1]
            if os.path.exists(maskfile):
                mask = Image.open(maskfile)
            else:
                mask = Image.new(mode='L', size=img.size, color=255)
            masks.append(mask)
        else:
            img, mask = align_face(imgfile)
            imgs.append(img)
            masks.append(mask)

    imgs_seq, latent_code = project(imgs, masks, **{arg_name: getattr(args, arg_name) for arg_name in arg_names})

    for i, input_name in enumerate(args.files):

        code_name = os.path.splitext(os.path.basename(input_name))[0] + ".pd"
        latent_file = {
            "latent_code": latent_code[i],
        }
        paddle.save(latent_file, code_name)

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        pil_img = Image.fromarray(imgs_seq[i][-1])
        pil_img.save(img_name)

        if args.save_mp4:
            fps = 30
            duration = 5
            save_video(
                imgs_seq[i],
                os.path.splitext(os.path.basename(input_name))[0] + "-project.mp4",
                fps, duration
            )
