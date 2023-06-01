
import numpy as np
import torch
import torch.nn as nn

from networks.blocks import ConvBNBlock, convspectralnorm_wrapper
from networks.activation import get_activation_obj, ParametricSoftplus
from networks.alexnet import Learner
import parser_utils as file_parser

def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m
if __name__ == "__main__":
    # torch.manual_seed(0)
    parser = file_parser.get_parser()
    args = parser.parse_args()

    model = Learner(
        n_inputs=(3, 32, 32),
        n_outputs=100,
        n_tasks=20,
        args=args
    )
    model_2 = Learner(
        n_inputs=(3, 32, 32),
        n_outputs=100,
        n_tasks=20,
        args=args
    )

    x = torch.rand(size=(64, 3, 32, 32))
    # x_ = torch.rand(size=(64, 3, 32, 32))
    
    output_1 = model(x)
    output_2 = model_2(x)

    dist_feature = torch.norm(model.feature_output - model_2.feature_output) / x.size(0)
    print(f"dist_feature: {dist_feature}")

    
    print(model.feature_output.shape)

    # total_params = flatten_params(m=model)
    # ones_vector = np.ones_like(total_params)

    # model = assign_weights(model, weights=ones_vector)
    
    # for name, p in model.named_parameters():
    #     print(f"name: {name}, p: {p}")
