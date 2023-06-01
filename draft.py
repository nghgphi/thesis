'''
if self.args.regularized_loss == 'grad':
            self.reg_const = self.args.reg_gnorm
            grad_x = torch.autograd.grad(loss, x, only_inputs=True, create_graph=True)[0]
            grad_norm = 1e5 * grad_x.pow(2).sum() / x.size(0)
            self.metadata['grad_norm'] = grad_norm.item()
            final_loss = raw_loss + self.reg_const * grad_norm
            return final_loss, self.metadata

        elif self.args.regularized_loss == 'curvature':
            self.reg_const = (self.args.reg_beta, self.args.reg_gamma)
            beta_term = 0.0
            for b in self.betas:
                beta_term += b
            if len(self.gammas) > 0:
                gamma_term = 0.
                for g in self.gammas:
                    gamma_term += g.abs()
            # self.metadata = {
            # 'raw_loss': raw_loss.item(),
            # 'beta_term': beta_term.item(),
            # 'gamma_term': gamma_term.item()
            # }
            self.metadata['raw_loss'] = raw_loss.item()
            self.metadata['beta_term'] = beta_term.item()
            self.metadata['gamma_term'] = gamma_term.item()
            if len(self.gammas) > 0:
                self.metadata.update({'gamma term:': gamma_term.item()})
                final_loss = loss + self.reg_const[0] * beta_term + self.reg_const[1] * gamma_term
            else:
                final_loss = loss + self.reg_const[0] * beta_term
            return final_loss, self.metadata
        elif self.args.regularized_loss == 'grad_curvature':
            self.reg_const = (self.args.reg_gnorm, self.args.reg_beta, self.args.reg_gamma)
            # curvature term
            beta_term = 0.0
            # print(f"self.betas: {self.betas.data.cpu()}")
            for idx, b in enumerate(self.betas):
                print(f"b_{idx} = {b.data.cpu()}")
                beta_term += b
            
            if len(self.gammas) > 0:
                gamma_term = 0.
                # print(f"self.gammas: {self.gammas.data.cpu()}")
                for idx, g in enumerate(self.gammas):
                    print(f"g_{idx} = {g.data.cpu()}")

                    gamma_term += g.abs()
                # print(f"")
            # grad norm term
            grad_x = torch.autograd.grad(loss, x, only_inputs=True, create_graph=True)[0]
            grad_norm = 1e5 * grad_x.pow(2).sum() / x.size(0)

            self.metadata = {
            'raw_loss': raw_loss.item(),
            'grad_norm': grad_norm.item(),
            'beta_term': beta_term.item(),
            'gamma_term': gamma_term.item()
            }
            if len(self.gammas) > 0:
                print(f"beta_term: {beta_term}, gamma_term: {gamma_term}")

                self.metadata.update({'gamma term:': gamma_term.item()})
                final_loss = loss + self.reg_const[0] * grad_norm + self.reg_const[1] * beta_term + self.reg_const[2] * gamma_term
            else:
                final_loss = loss + self.reg_const[0] * grad_norm + self.reg_const[1] * beta_term
            return final_loss, self.metadata
        else:
            return loss/len(y), self.metadata
'''