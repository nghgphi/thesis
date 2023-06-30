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
        self.lambda_gp = args.lambda_gp
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
                'loss_old': 0.0,
                'loss_new': 0.0,
                'loss_at_new': 0.0,
                'loss_at_old': 0.0,
                'gradient_penalty': 0.0
            }
            # print(x_n.shape)
            
            # _____________loss CE on new data => SAM___________________
            loss_new = self.compute_loss_with_SAM(x_n, y_n, t_n, use_SAM=True)
            metadata['loss_new'] = loss_new.item()
            
            # _____________loss AT on new data: => no SAM_______________
            delta_n = self.pgd_linf(x_n, y_n, t_n, buffer=False)
            x_n_at = x_n + delta_n
            loss_at_new = self.compute_loss_with_SAM(x_n_at, y_n, t_n, use_SAM=False)
            metadata['loss_at_new'] = loss_at_new.item()
                # total_loss = loss_new + loss_old + self.lambda_at * loss_at_old + self.lambda_at * loss_at_new  + self.lambda_gp * gradient_penalty
                # total_loss = loss_new + loss_old + self.lambda_at * loss_at_old + self.lambda_at * loss_at_new  
            # else:
                # total_loss = loss_new 
                # total_loss = loss_new + self.lambda_at * loss_at_new
            sum_loss_new = self.lambda_at * loss_at_new + loss_new
            sum_loss_new.backward()
            # print(f'metadata: {metadata}')
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            if len(self.M_vec) > 0:
                argmin_idx = self.pairing([x_n, y_n], [x_o, y_o])
                print(f'argmin_idx: {argmin_idx.shape}')

                # update the lambdas
                if self.args.method in ['dgpm', 'xdgpm']:
                    torch.nn.utils.clip_grad_norm_(self.lambdas.parameters(), self.args.grad_clip_norm)
                    if self.args.sharpness:
                        self.opt_lamdas.step()
                    else:
                        self.opt_lamdas_step()

                    for idx in range(len(self.lambdas)):
                        self.lambdas[idx] = nn.Parameter(torch.sigmoid(self.args.tmp * self.lambdas[idx]))
                # train on the rest of subspace spanned by GPM
                self.train_restgpm()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()

                self.zero_grads()

                

                # print(f'self.current_task: {self.current_task}')
                if len(x_o) > 0:
                    # _______________loss CE on buffer: => SAM________________
                    loss_old = self.compute_loss_with_SAM(x_n_at, y_n, t_n, use_SAM=True)
                    metadata['loss_old'] = loss_old.item()

                    # _______________loss AT on buffer: => no SAM_____________
                    # x_o_at = self.fgsm_attack(x_o, y_o, t_o, 0.07, buffer=True)
                    # print(f't_o: {type(t_o)} | {t_o.data}')
                    delta_o = self.pgd_linf(x_o, y_o, t_o, buffer=True)  
                    x_o_at = x_o + delta_o
                    loss_at_old = self.compute_loss_with_SAM(x_n_at, y_n, t_n, use_SAM=False)
                    metadata['loss_at_old'] = loss_at_old.item()

                    # ________________INTERPOLATE BWT BUFFER AND BUFFER AT => NO SAM________
                    # if len(x_o.shape) == 2:
                    #     alpha = torch.Tensor(np.random.random((x_o_at.size(0), 1))).cuda()
                    # elif len(x_o.shape) == 4:
                    #     alpha = torch.Tensor(np.random.random((x_o_at.size(0), 1, 1, 1))).cuda()
                    # # Get random interpolation between real and fake samples
                    
                    # interpolates = alpha * x_o + ((1 - alpha) * x_o_at)
                    # # print(f'alpha: {alpha.shape} | x_o: {x_o.shape} --- x_o_at: {x_o_at.shape} | interpolates: {interpolates.shape}')

                    # interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

                    # output = self.net.forward(interpolates)
                    # gradients = torch.autograd.grad(
                    #                                     output,
                    #                                     interpolates,
                    #                                     grad_outputs=torch.ones((interpolates.shape[0], self.net.n_outputs)).cuda(),
                    #                                     create_graph=True,
                    #                                     retain_graph=True,
                    #                                     only_inputs=True,
                    # )[0]
                    # gradients = gradients.view(gradients.size(0), -1)
                    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    # metadata['gradient_penalty'] = gradient_penalty

                sum_loss_old = loss_old + self.lambda_at * loss_at_old 
                sum_loss_old.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
            else:
                self.optimizer.step()
                
                
                self.zero_grads()

                delta_n = self.pgd_linf(x_n, y_n, t_n, buffer=False)
                x_n_at = x_n + delta_n
                loss_at_new = self.compute_loss_with_SAM(x_n_at, y_n, t_n, use_SAM=False)
                metadata['loss_at_new'] = loss_at_new.item()

                loss_at_new.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()

            if self.real_epoch == 0:
                self.push_to_mem(x, y, torch.tensor(t))
            self.zero_grads()

                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch
        
        # print(f"metadata['reg_term']: {metadata['reg_term']}")
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