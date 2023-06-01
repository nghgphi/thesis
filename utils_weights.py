import numpy as np

import torch
import torch.nn as nn


def flatten_params(m, numpy_output=True):
    total_params = []
    for name, param in m.named_parameters():
        # print(f"name: {name}, param:{param.shape}")
        if 'lambda' not in name:
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    # print(f"Total params: {len(total_params)}")

    if numpy_output:
        return total_params.cpu().detach().numpy()
    # print(f"Total params: {len(total_params)}")
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    # print(f"state_dict: {state_dict.keys()}")
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param or 'lambda' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count

    print(f"Total assigned weights / len(weights) = {index}/{len(weights)} ")
    m.load_state_dict(state_dict)
    return m