import torch
import os

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def checkpoint_state(model=None, optimizer=None, epoch=None, other_state=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'other_state': other_state}

def save_checkpoint(state, filename):
    torch.save(state, filename)

def find_match_key(key, dic):
    # key: pretrained model key
    for _k in dic.keys():
        if key == _k:
            return _k
    key = '.'.join(key.split('.')[1:])
    if key.split('.')[0] in ['fea_compression']:
        split_point = 0
    else:
        split_point = 1
    for _k, _ in dic.items():
        k = '.'.join(_k.split('.')[split_point:])
        if key == k:
            return _k
    return None

"""def load_pretrained_model(model, filename, to_cpu=False, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from pre-trained checkpoint {} to {}'.format(filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    if checkpoint.get('model_state', None) is not None:
        checkpoint = checkpoint.get('model_state')
    
    update_model_state = {}
    for key, val in checkpoint.items():
        match_key = find_match_key(key, model.state_dict())
        if match_key is None:
            print("Cannot find a matched key for {}".format(key))
            continue
        if model.state_dict()[match_key].shape == checkpoint[key].shape:
            update_model_state[match_key] = val

    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    for key in state_dict:
        if key not in update_model_state:
            logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
"""
def load_pretrained_model(model, filename,  to_cpu=False, logger=None):

    if not os.path.isfile(filename):
        raise FileNotFoundError

    print("LOAD CHECKPOINT FROM : ", filename)
    my_model_dict = model.state_dict()
    if to_cpu == False:
        pre_weight = torch.load(filename, map_location='cuda:'+str(device))
    else:
        pre_weight = torch.load(filename)
    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        elif k in my_model_dict and my_model_dict[k].shape == value.permute(4,0,1,2,3).shape:
            match_size += 1
            part_load[k] = value.permute(4,0,1,2,3)
        elif k in my_model_dict and k.split('.')[-2] == 'conv3':
            match_size += 1
            part_load[k] = value[0].unsqueeze(0).permute(4,0,1,2,3)
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)


def load_params_with_optimizer(model, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except:
                logger.info('Optimizer could not be loaded.')

    logger.info('==> Done')

    return epoch

def load_params_with_optimizer_otherstate(model, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except:
                logger.info('Optimizer could not be loaded.')

    other_state = checkpoint.get('other_state', None)

    logger.info('==> Done')

    return epoch, other_state

def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.OPTIMIZE.LR)
    return optimizer

def build_scheduler(optimizer, cfg, last_epoch):
    return None
