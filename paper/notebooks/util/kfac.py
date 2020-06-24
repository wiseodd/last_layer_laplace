##########################################################################
#
#  Taken with modifications from
#  https://github.com/wjmaddox/swa_gaussian/
#
##########################################################################


import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


class KFAC(Optimizer):

    def __init__(self, net, alpha=0.95):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            alpha (float): Running average parameter (if == 1, no r. ave.).
        """
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0

        for mod in net.modules():
            mod_class = mod.__class__.__name__

            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)

                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

                params = [mod.weight]

                if mod.bias is not None:
                    params.append(mod.bias)

                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)

        super(KFAC, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None

            state = self.state[group['mod']]
            self._compute_covs(group, state)

        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()

        # if mod.bias is not None:
        #     ones = torch.ones_like(x[:1])
        #     x = torch.cat([x, ones], dim=0)

        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))

        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1

        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))
