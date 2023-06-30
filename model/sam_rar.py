import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import *

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        
        self.eta2 = args.eta2
        self.args = args
        self.lambda_at = args.lambda_at
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
    def fgsm_attack(self, images, labels, tasks, eps, buffer=False) :

        # print(images.requires_grad)
        self.net.eval()
        
        # with torch.no_grad():
        images_ = images.detach()
        
        # for _ in range(1):
            # images_ = images_.detach()
        images_.requires_grad = True

        outputs = self.net(images_)
        
        self.net.zero_grad()

        if not buffer:
            cost = self.loss_criterion(outputs, labels)
        else:
            cost = self.meta_loss(images_, labels, tasks)

        cost.backward()
        
        images_ = images_ + eps * images_.grad.sign()
        images_ = torch.clamp(images_, 0, 1)
        
        self.net.train()
        return images_
    
    def pgd_linf(self, X, y, t, epsilon=0.1, alpha=0.01, num_iter=2, buffer=True):
        """ Construct FGSM adversarial examples on the examples X"""
        delta = torch.zeros_like(X, requires_grad=True)
            
        for it in range(num_iter):
            if not buffer:
                loss = nn.CrossEntropyLoss()(self.net(X + delta), y)
            else:
                loss = self.meta_loss(X + delta, y, t)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        return delta.detach()
    
    def compute_loss_with_SAM(self, x, y, t, use_SAM=True):
        fast_weights = None
        if use_SAM:
            fast_weights = self.compute_w_adv(x, y, t, fast_weights)
        loss = self.meta_loss(x, y, t, fast_weights)

        return loss

    def observe(self, x, y, t):
        if t != self.current_task:
            self.current_task = t
        self.net.train()

        reg_terms = []
        for pass_itr in range(self.glances):
            self.iter += 1
            self.zero_grads()

            perm = torch.randperm(x.size(0))

            x = x[perm]
            y = y[perm]
            
            (x_n, y_n, t_n), (x_o, y_o, t_o) = self.get_batch(x, y, t)  # 
            
            metadata = {
                'loss_theta^': 0.0
            }
            
            # update model => theta^' - Eq(5)
            
            loss = self.compute_loss_with_SAM(x_n, y_n, t_n, use_SAM=True)
            loss.backward()
            metadata['loss_theta^'] = loss.item()

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
                self.train_restgpm()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
            else:
                self.optimizer.step()
                
            if self.real_epoch == 0:
                self.push_to_mem(x, y, torch.tensor(t))
            
            self.zero_grads()

            # select new example corresponding to each old example - Eq(9)

            if len(x_o) > 0:
                _ = self.net.forward(x_n)
                feat_new = self.net.feat_out
                
                _ = self.net.forward(x_o)
                feat_old = self.net.feat_out

                print(f'feat_new: {feat_new.shape} - feat_out: {feat_old.shape}')

                argmin_idx = self.pairing([feat_new, y_n], [feat_old, y_o])

        return metadata
    def compute_w_adv(self, x, y, t, fast_weights):
        loss = self.meta_loss(x, y, t, fast_weights)
        
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
    def pairing(self, new_data, buffer):
        '''
            For each example in buffer, find the example with the different label and nearest to the this buffer's example
        '''
        x_new, y_new = new_data[0], new_data[1]
        x_old, y_old = buffer[0], buffer[1]

        x_new_ex = x_new.unsqueeze(0)
        x_old_ex = x_old.unsqueeze(1)
        dist = F.pairwise_distance(x_old_ex, x_new_ex)

        mask = y_old.unsqueeze(1) != y_new.unsqueeze(0)
        dist_masked = torch.where(mask, dist, float('inf'))

        _, argmin_idx = torch.min(dist_masked, dim=1)

        return argmin_idx