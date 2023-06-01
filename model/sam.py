import math
import numpy as np

import torch
import torch.nn as nn

from model.base import *

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        
        self.eta2 = args.eta2
        self.eta3 = args.eta3
        self.args = args
        self.batch_size = args.batch_size
        
        self.update_optimizer()
        print(f"Total parameters need to be updated:{self.count_parameters()}")
        print(f"len(param_group): {len(self.optimizer.param_groups)}")

        # self.param_groups = self.optimizer.param_groups
    def forward(self, x, t):
        output = self.net.forward(x)

        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)

        return output

    def observe(self, x, y, t, one_task_only=False, copy_net=None):
        if t != self.current_task:
            self.current_task = t
        for pass_itr in range(self.glances):
            self.iter += 1
            self.zero_grads()

            perm = torch.randperm(x.size(0))

            x = x[perm]
            y = y[perm]
            
            bx, by, bt = self.get_batch(x, y, t)  # from a batch, recreate a new batch from this and some example from old examples

            fast_weights = None
            fast_weights = self.compute_w_adv(x, y, t, fast_weights)
            if one_task_only:
                loss = self.take_loss(x, y, t, fast_weights)

                if copy_net is not None:
                    # print("Not None")
                    outputs = copy_net.forward(x, fast_weights)
                    copy_net_feature = copy_net.feature_output
                    main_net_feature = self.net.feature_output

                    assert copy_net_feature.shape == main_net_feature.shape, f'Not the same shape: copy_net_feature: {copy_net_feature.shape} != main_net_feature: {main_net_feature.shape}'
                    dist = 1.0 / x.size(0) * torch.norm(main_net_feature - copy_net_feature.detach())
                    loss += self.eta3 * dist # distill 
                    self.dist = dist.item()
                    # print(self.dist)
                self.zero_grads()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                metadata = {
                    'loss': loss,
                    'cur_grad': 0.0,
                    'prev_grad': 0.0
                }
            else:
                metadata = self.meta_loss(bx, by, bt, fast_weights)
                loss = metadata['loss']
                self.zero_grads()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

                if len(self.M_vec) > 0:

                    # update the lambdas
                    if self.args.method in ['dgpm', 'xdgpm']:
                        torch.nn.utils.clip_grad_norm_(self.lambdas.parameters(), self.args.grad_clip_norm)
                        if self.args.sharpness:
                            self.opt_lamdas.step()
                        else:
                            self.opt_lamdas_step()

                        for idx in range(len(self.lambdas)):
                            self.lambdas[idx] = nn.Parameter(torch.sigmoid(self.args.tmp * self.lambdas[idx]))

                    # only use updated lambdas to update weight
                    

                    # train on the rest of subspace spanned by GPM
                    self.train_restgpm()
                    # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                    self.optimizer.step()

                else:
                    self.optimizer.step()

                # self.zero_grads()

                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch 
            if self.real_epoch == 0:
                self.push_to_mem(x, y, torch.tensor(t))

        return metadata
    def compute_w_adv(self, x, y, t, fast_weights):
        loss = self.take_loss(x, y, t, fast_weights)
        
        if fast_weights is None:
            fast_weights = self.net.get_params()
        
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))
        
        grad_norm = torch.norm(torch.stack(
            [
                torch.norm(p, p=2) for p in grads if p is not None
            ]
        ))
        for i in range(len(grads)):
            if grads[i] is not None:
                grads[i] = torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)

        if self.args.sharpness:
            fast_weights = list(map(lambda p: p[1] + self.args.rho * p[0] / (grad_norm + 1e-12) if p[0] is not None else p[1], zip(grads, fast_weights)))

        return fast_weights

    # def w_adv(self, x, y, t, fast_weights):

    #     loss = self.take_loss(x, y, t, fast_weights) # loss of current batch

    #     self.zero_grads()
    #     loss.backward()

    #     # grad_norm = torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=self.args.grad_clip_norm)
    #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

    #     grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.net.parameters()]), p=2)

    #     if fast_weights is None:
    #         fast_weights = self.net.get_params()

        
        
    #     # print(f"len(fast_weights): {len(fast_weights)}")

    #     graph_required = self.args.second_order 
    #     param_groups = self.optimizer.param_groups
    #     grads = []
    #     # grads = list(torch.autograd.grad(loss, fast_weights, allow_unused=True, retain_graph=graph_required, create_graph=graph_required))
    #     grads = list(p.grad for group in param_groups for p in group["params"]  if p.requires_grad is True)
    #     assert len(fast_weights) == len(grads), f"Two lists are not the same length ({len(fast_weights)} != {len(grads)})"

    #     for i in range(len(grads)):
    #         if grads[i] is not None:
    #             grads[i] = torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)

    #     # w_adv = w + rho * grad / (grad_norm + eps)
    #     # print(f"len fast_weight: {len(fast_weights)}, grads: {len(grads)}\n" + "_" * 50)

    #     for i in range(len(fast_weights)):
    #         # print(f"shape fast_weight: {fast_weights[i].shape}, grads: {grads[i].shape}")
    #         fast_weights[i].data += self.args.rho * grads[i] / (grad_norm + 1e-7)
    #     self.net.zero_grad()
    #     return fast_weights

    # def compute_grad_norm(self, x, y, t, fast_weights):
    #     loss = self.meta_loss(x, y, t, fast_weights) # loss of current batch

    #     with torch.autograd.set_detect_anomaly(True):
    #         self.zero_grads()
    #         loss.backward()

    #         # grad_norm = torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=self.args.grad_clip_norm)
    #         torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

    #         grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.net.parameters()]), p=2)

    #         # print(f"grad_norm: {grad_norm}")
    #     return grad_norm


    def zero_grads(self):
        self.optimizer.zero_grad()
        self.net.zero_grad()
        if len(self.M_vec) > 0 and self.args.method in ['dgpm', 'xdgpm']:
            self.lambdas.zero_grad()
    def define_lambda_params(self):
        assert len(self.M_vec) > 0

        # Setup learning parameters
        self.lambdas = nn.ParameterList([])
        for i in range(len(self.M_vec)):
            self.lambdas.append(nn.Parameter(self.args.lam_init * torch.ones((self.M_vec[i].shape[1]), requires_grad=True)))

        if self.cuda:
            self.lambdas = self.lambdas.cuda()

        return
    def update_opt_lambda(self, lr=None):
        if lr is None:
            lr = self.eta2
        self.opt_lamdas = torch.optim.SGD(list(self.lambdas.parameters()), lr=lr, momentum=self.args.momentum)

        return

    def opt_lamdas_step(self):
        """
            Performs a single optimization step, but change gradient descent to ascent
            """
        for group in self.opt_lamdas.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.opt_lamdas.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data = (p.data + group['lr'] * d_p).clone()

        return
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters())








                

                

