import torch
import torchfile


def get_module(obj, idx):
    for i in idx:
        obj = obj[b'modules'][i]
    return obj


def conv_3d(py_dest, lua_source):
    state_dict = {
        'weight': torch.from_numpy(lua_source[b'weight']),
        'bias': torch.from_numpy(lua_source[b'bias'])
    }
    py_dest.load_state_dict(state_dict)


def batch_norm_3d(py_dest, lua_source):
    state_dict = {
        'weight': torch.from_numpy(lua_source[b'weight']),
        'bias': torch.from_numpy(lua_source[b'bias']),
        'running_mean': torch.from_numpy(lua_source[b'running_mean']),
        'running_var': torch.from_numpy(lua_source[b'running_var'])
    }
    py_dest.load_state_dict(state_dict)


def basic_3d_block(py_dest, lua_source):
    conv_3d(py_dest.block[0], get_module(lua_source, [0]))
    batch_norm_3d(py_dest.block[1], get_module(lua_source, [1]))


def res_branch(py_dest, lua_source):
    conv_3d(py_dest[0], get_module(lua_source, [0]))
    batch_norm_3d(py_dest[1], get_module(lua_source, [1]))
    conv_3d(py_dest[3], get_module(lua_source, [3]))
    batch_norm_3d(py_dest[4], get_module(lua_source, [4]))


def skip_con(py_dest, lua_source):
    if len(py_dest) > 0:
        conv_3d(py_dest[0], get_module(lua_source, [0]))
        batch_norm_3d(py_dest[1], get_module(lua_source, [1]))


def res_3d_block(py_dest, lua_source):
    res_branch(py_dest.res_branch, get_module(lua_source, [0, 0]))
    skip_con(py_dest.skip_con, get_module(lua_source, [0, 1]))


def conv_transpose_3d(py_dest, lua_source):
    state_dict = {
        'weight': torch.from_numpy(lua_source[b'weight']),
        'bias': torch.from_numpy(lua_source[b'bias'])
    }
    py_dest.load_state_dict(state_dict)


def upsample_3d_block(py_dest, lua_source):
    conv_transpose_3d(py_dest.block[0], get_module(lua_source, [0]))
    batch_norm_3d(py_dest.block[1], get_module(lua_source, [1]))


def load_lua_weights(model, path):
    lua_v2v = torchfile.load(path)

    # Front Layers
    basic_3d_block(model.front_layers[0], get_module(lua_v2v, [0]))
    res_3d_block(model.front_layers[2], get_module(lua_v2v, [2]))
    res_3d_block(model.front_layers[3], get_module(lua_v2v, [3]))
    res_3d_block(model.front_layers[4], get_module(lua_v2v, [4]))

    # Encoder/Decoder
    res_3d_block(model.encoder_decoder.encoder_res1, get_module(lua_v2v, [5, 0, 0, 1]))
    res_3d_block(model.encoder_decoder.encoder_res2, get_module(lua_v2v, [5, 0, 0, 2, 0, 1]))
    res_3d_block(model.encoder_decoder.mid_res, get_module(lua_v2v, [5, 0, 0, 2, 0, 2]))
    res_3d_block(model.encoder_decoder.decoder_res2, get_module(lua_v2v, [5, 0, 0, 2, 0, 3]))
    upsample_3d_block(model.encoder_decoder.decoder_upsample2, get_module(lua_v2v, [5, 0, 0, 2, 0, 4]))
    res_3d_block(model.encoder_decoder.decoder_res1, get_module(lua_v2v, [5, 0, 0, 4]))
    upsample_3d_block(model.encoder_decoder.decoder_upsample1, get_module(lua_v2v, [5, 0, 0, 5]))
    res_3d_block(model.encoder_decoder.skip_res1, get_module(lua_v2v, [5, 0, 1]))
    res_3d_block(model.encoder_decoder.skip_res2, get_module(lua_v2v, [5, 0, 0, 2, 1]))

    # Back layers
    res_3d_block(model.back_layers[0], get_module(lua_v2v, [5, 2]))
    basic_3d_block(model.back_layers[1], get_module(lua_v2v, [5, 3]))
    basic_3d_block(model.back_layers[2], get_module(lua_v2v, [5, 4]))

    # Output layer
    conv_3d(model.output_layer, get_module(lua_v2v, [6]))