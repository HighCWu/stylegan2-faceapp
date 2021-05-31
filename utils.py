import inspect
import numpy as np
import moviepy.editor as mpy

import paddle
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url
from ppgan.models.generators import StyleGANv2Generator as Generator
from ppgan.models.generators import Pixel2Style2Pixel

LMS_URL = 'https://paddlegan.bj.bcebos.com/models/lms.dat'


generator_model_cfgs = {
    'ffhq-config-f': {
        'model_urls': 'https://paddlegan.bj.bcebos.com/models/stylegan2-ffhq-config-f.pdparams',
        'size': 1024,
        'style_dim': 512,
        'n_mlp': 8,
        'channel_multiplier': 2
    }
}


pSp_model_cfgs = {
    'ffhq-inversion': {
        'model_urls': 'https://paddlegan.bj.bcebos.com/models/pSp-ffhq-inversion.pdparams',
        'size': 1024,
        'style_dim': 512,
        'n_mlp': 8,
        'channel_multiplier': 2
    }
}


class AttrDict(dict):
    def __getattr__(self, key):
        # return self[key]
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


__arg_types__ = []
__kwarg_types__ = {}
def arg_type(base_cls, *args, **kwargs):
    if hasattr(base_cls, '__args__'):
        base_type = base_cls.__args__[0]
        nargs = True
    else:
        base_type = base_cls
        nargs = False
        
    cls_sig = None
    if isinstance(base_cls, str):
        cls_sig = base_cls
        base_cls = object
        base_type = bool

    class ChildCls(base_cls):
        __type__ = base_type
        __args__ = list(args)
        __kwargs__ = kwargs
        __nargs__ = nargs

    __arg_types__.append(ChildCls)

    if cls_sig is not None:
        __kwarg_types__[cls_sig] = ChildCls
        return bool

    return ChildCls
    

def func_args(parser, func):
    sig = inspect.signature(func)
    arg_names = []
    for arg_name, arg_attrs in sig.parameters.items():
        arg_cls = arg_attrs.annotation
        type_sig = func.__name__ + ':' + arg_name
        if arg_cls == bool and type_sig in __kwarg_types__:
            arg_cls = __kwarg_types__[type_sig]
        if arg_cls in __arg_types__:
            use_default = not isinstance(arg_attrs.default, inspect._empty)
            args = ['--'+arg_name] + arg_cls.__args__
            kwargs = dict(**arg_cls.__kwargs__)
            if not 'action' in kwargs:
                kwargs['type'] = arg_cls.__type__
            if use_default:
                if not 'action' in kwargs:
                    kwargs['default'] = arg_attrs.default
            else:
                kwargs['required'] = True
            if arg_cls.__nargs__:
                kwargs['nargs'] = '+'

            parser.add_argument(*args, **kwargs)
            arg_names.append(arg_name)
        
    return parser, arg_names


def save_video(images, filename, fps, duration):
    def make_frame(t):
        idx = min(int(np.ceil(t / duration * len(images))), len(images)-1)
        frame = images[idx]

        return frame
    
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_videofile(filename, fps=fps)


def get_generator(
    weight_path=None,
    model_type='ffhq-config-f',
    size=1024,
    style_dim=512,
    n_mlp=8,
    channel_multiplier=2
):
    if weight_path is None:
        if model_type in generator_model_cfgs.keys():
            weight_path = get_weights_path_from_url(generator_model_cfgs[model_type]['model_urls'])
            size = generator_model_cfgs[model_type].get('size', size)
            style_dim = generator_model_cfgs[model_type].get('style_dim', style_dim)
            n_mlp = generator_model_cfgs[model_type].get('n_mlp', n_mlp)
            channel_multiplier = generator_model_cfgs[model_type].get('channel_multiplier', channel_multiplier)
            checkpoint = paddle.load(weight_path)
        else:
            raise ValueError('Predictor need a weight path or a pretrained model type')
    else:
        checkpoint = paddle.load(weight_path)

    generator = Generator(size, style_dim, n_mlp, channel_multiplier)
    generator.set_state_dict(checkpoint)

    return generator


def get_pSp(
    weight_path=None,
    model_type='ffhq-inversion',
    size=1024,
    style_dim=512,
    n_mlp=8,
    channel_multiplier=2
):
    if weight_path is None:
        if model_type in pSp_model_cfgs.keys():
            weight_path = get_weights_path_from_url(pSp_model_cfgs[model_type]['model_urls'])
            size = pSp_model_cfgs[model_type].get('size', size)
            style_dim = pSp_model_cfgs[model_type].get('style_dim', style_dim)
            n_mlp = pSp_model_cfgs[model_type].get('n_mlp', n_mlp)
            channel_multiplier = pSp_model_cfgs[model_type].get('channel_multiplier', channel_multiplier)
            checkpoint = paddle.load(weight_path)
        else:
            raise ValueError('Predictor need a weight path or a pretrained model type')
    else:
        checkpoint = paddle.load(weight_path)

    opts = checkpoint.pop('opts')
    opts = AttrDict(opts)
    _opts = AttrDict(
        size=size,
        style_dim=style_dim,
        n_mlp=n_mlp,
        channel_multiplier=channel_multiplier
    )
    opts.update(_opts)

    pSp = Pixel2Style2Pixel(opts)
    pSp.set_state_dict(checkpoint)

    return pSp
