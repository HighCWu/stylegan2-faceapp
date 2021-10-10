'''Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/closed_form_factorization.py and '''
'''         https://github.com/rosinality/stylegan2-pytorch/blob/master/apply_factor.py'''

import numpy as np
import paddle

from utils import arg_type, func_args, make_image, get_generator

from PIL import Image
from tqdm import tqdm


svd_vector = None
def get_factorized_style_vector(generator, index):
    global svd_vector
    if svd_vector is None:
        named_parameters = dict(generator.named_parameters())
        modulate = {
            k: v
            for k, v in named_parameters.items()
            if "modulation" in k and "to_rgbs" not in k and "weight" in k
        }

        weight_mat = []
        for _, v in modulate.items():
            weight_mat.append(v.t())

        W = paddle.concat(weight_mat, 0)
        svd_vector = paddle.to_tensor(np.linalg.svd(W.numpy())[-1].T.astype('float32'))

    return svd_vector[:,index].unsqueeze(0).unsqueeze(0).tile([1,1,1])


def edit(
    latent_codes: paddle.Tensor = None,
    generator = None,
    ckpt: arg_type(str, help="path to the model checkpoint") = None,
    model_type: arg_type(str, help="inner model type. `ffhq-config-f` for default genrator and `ffhq-inversion` for pSp") = None,
    size: arg_type(int, help="original output image resolution") = 1024,
    style_dim: arg_type(int, help="dimensions of style z") = 512,
    n_mlp: arg_type(int, help="the number of multi-layer perception layers for style z") = 8,
    channel_multiplier: arg_type(int, help="channel product, affect model size and the quality of generated pictures") = 2,
    style_vector_file: arg_type(str, help="path to the style vector") = 'docs/stylegan2directions/age.npy',
    start_offset: arg_type(float, help="the first offset of the style vector relative to the latent code") = -5,
    final_offset: arg_type(float, help="the last offset of the style vector relative to the latent code") = 5,
    offset_step_size: arg_type(float, help="the offset step size from the start offset to the final offset") = 0.1,
    use_factorizer: arg_type(
       'edit:use_factorizer', action="store_true", 
       help="whether to use the style vectors of the factorizer"
    ) = False,
    factorizer_style_index: arg_type(int, help="the index of one style vector in the style vectors of the factorizer") = 0,
):
    
    generator = generator if generator is not None else get_generator(
        weight_path=None if ckpt is None else ckpt,
        model_type='ffhq-config-f' if model_type is None else model_type,
        size=size,
        style_dim=style_dim,
        n_mlp=n_mlp,
        channel_multiplier=channel_multiplier
    )
    
    if use_factorizer:
        style_vector = get_factorized_style_vector(generator, factorizer_style_index)
    else:
        style_vector = paddle.to_tensor(np.load(style_vector_file)).astype('float32').unsqueeze(0)
        
    degrees = []
    degree = start_offset
    while degree <= final_offset + 1e-8:
        degrees.append(degree)
        degree += offset_step_size

    frames = []
    latent_codes_seqs = []
    for degree in tqdm(degrees):
        with paddle.no_grad():
            latent_in = latent_codes + degree * style_vector
            img_gen, _ = generator([latent_in], input_is_latent=True, randomize_noise=False)
        frames.append(make_image(img_gen))
        latent_codes_seqs.append(latent_in)

    imgs_seq = [[] for _ in range(img_gen.shape[0])]
    latents_seq = [[] for _ in range(latent_in.shape[0])]
    for i in range(img_gen.shape[0]):
        for frame, latent_codes_seq in zip(frames, latent_codes_seqs):
            imgs_seq[i].append(frame[i])
            latents_seq[i].append(latent_codes_seq[i])

    return imgs_seq, latents_seq


if __name__ == "__main__":
    import argparse
    import os

    from utils import save_video

    parser = argparse.ArgumentParser(
        description="Image attributes editor in stylegan2 latent space"
    )
    parser, arg_names = func_args(parser, edit)
    parser.add_argument(
        "--save_mp4", action="store_true", help="saving training progress images as mp4 videos"
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image latent codes to be edited"
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="output directory"
    )

    args = parser.parse_args()

    latent_codes = []
    for path in args.files:
        latent_code = paddle.load(path)['latent_code']
        latent_codes.append(latent_code)
    latent_codes = paddle.stack(latent_codes, 0)

    imgs_seq, latents_seq = edit(latent_codes, **{arg_name: getattr(args, arg_name) for arg_name in arg_names})

    os.makedirs(args.output, exist_ok=True)
    for i, input_name in enumerate(args.files):

        for j, latent_code in enumerate(latents_seq[i]):
            code_name = os.path.join(
                args.output, 
                os.path.splitext(os.path.basename(input_name))[0] + "-" + str(j).zfill(3) + ".pd"
            )
            latent_file = {
                "latent_code": latent_code,
            }
            paddle.save(latent_file, code_name)

        img_name = os.path.join(
            args.output, 
            os.path.splitext(os.path.basename(input_name))[0] + "-start-offset.png"
        )
        pil_img = Image.fromarray(imgs_seq[i][0])
        pil_img.save(img_name)

        img_name = os.path.join(
            args.output, 
            os.path.splitext(os.path.basename(input_name))[0] + "-final-offset.png"
        )
        pil_img = Image.fromarray(imgs_seq[i][-1])
        pil_img.save(img_name)

        if args.save_mp4:
            fps = 30
            duration = 5
            save_video(
                imgs_seq[i],
                os.path.join(
                    args.output, 
                    os.path.splitext(os.path.basename(input_name))[0] + "-offset.mp4"
                ),
                fps, duration
            )
