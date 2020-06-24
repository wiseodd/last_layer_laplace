import torch
from torch import nn, autograd
from tqdm import tqdm, trange
import numpy as np
from math import *
from hessian import *
import torch.nn.functional as F
import torch.distributions as dist
from backpack import backpack, extend, extensions
from scipy.linalg import block_diag
from backpack.hessianfree.ggnvp import ggn_vector_product


class DiagLaplace(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    """

    def __init__(self, base_model):
        super().__init__()

        self.net = type(base_model)()
        self.net.load_state_dict(base_model.state_dict())
        self.net.eval()
        self.params = []
        self.net.apply(lambda module: laplace_parameters(module, self.params))
        self.hessian = None

    def forward(self, x):
        return self.net.forward(x)
            
#     def estimate_variance_batch(self, X, y, var0):
#         tau = 1/var0
        
#         params = torch.cat([p.flatten() for p in self.net.parameters()])
#         n = len(params)

#         nll = F.binary_cross_entropy_with_logits(self(X).squeeze(), y, reduction='sum')
#         loss = nll + 1/2 * params @ (tau*torch.eye(n)) @ params
#         h = exact_hessian(loss, self.net.parameters()).detach()

#         self.hessian = torch.inverse(h + torch.eye(h.shape[0]))
# #             print(torch.symeig(self.hessian)[0][:10])

    def estimate_variance_batch(self, X, y, var0, multiclass=False):
        tau = 1/var0
        
        params = torch.cat([p.flatten() for p in self.net.parameters()])
        n = len(params)

        f = self(X).squeeze()
        
        output = self(X).squeeze()
        
        if not multiclass:
            nll = F.binary_cross_entropy_with_logits(output, y, reduction='sum')
        else:
            nll = F.cross_entropy(output, y, reduction='sum')
            
        loss = nll
#         loss = nll + 1/2 * params @ (tau*torch.eye(n)) @ params
        
        num_params = sum(p.numel() for p in self.net.parameters())
        ggn = torch.zeros(num_params, num_params)
        
        for i in range(num_params):
            v = torch.zeros(num_params)
            v[i] = 1.
            v = vector_to_parameter_list(v, self.net.parameters())
            ggn_i = torch.cat([a.flatten() for a in ggn_vector_product(loss, output, self.net, v)])
            ggn[i, :] = ggn_i
        
        h = ggn.detach()
        
#         h = exact_hessian(loss, self.net.parameters()).detach()

        self.hessian = torch.inverse(h + tau*torch.eye(num_params))        
        
    def forward_linearized(self, x, sigm=True, progress=True):
        # MAP output
        inputs = self.net.parameters()
        f_map = self(x)
        
        # Gradient. Let N be num of data, P be num of paramsf
        d = []
        pbar = trange(len(x), position=0, leave=True) if progress else range(len(x))
        for i in pbar:
            d_ = autograd.grad([f_map[i]], self.net.parameters(), retain_graph=True)
            d_flat = torch.cat([a.flatten() for a in d_])
            d.append(d_flat)  # (P,)
        d = torch.stack(d)  # (N, P)
            
        f_map = f_map.flatten()
        d = d  
        
        # Hessian
        h = self.hessian  # (P, P)
                        
        # z
        denom = torch.sqrt(1 + pi/8 * torch.diag(d @ h @ d.t()))
        z = f_map/denom
                        
        return torch.sigmoid(z) if sigm else z
    
    
    def forward_linearized_multi(self, x, progress=True):
        # MAP output
        inputs = self.net.parameters()
        f_map = self(x)
        
        # Jacobian. Let N be num of data, P be num of params, K num of classes
        J = []
        pbar = trange(len(f_map), position=0, leave=True) if progress else range(len(x))
        for i in pbar:
            J_ = []
            for j in range(len(f_map[i])):
                d_ = grad([f_map[i, j]], self.net.parameters(), retain_graph=True)
                d_flat = torch.cat([a.flatten() for a in d_])
                J_.append(d_flat)
            J.append(torch.stack(J_))
        J = torch.stack(J)
            
        f_map = f_map.detach()  # (N, K)
        J = J.detach()  # (N, K, P)
                
        # Hessian
        H = self.hessian[None, :, :]  # (1, P, P)
        Cov = J @ H @ J.transpose(1, 2)  # (N, K, P) x (1, P, P) x (N, P, K) -> (N, K, K)
                
        N_f = dist.MultivariateNormal(f_map, Cov)  # N K-dim MVN
                
        # MC Integral
        py = 0
        for _ in range(1000):
            f_s = N_f.rsample()
            py += torch.softmax(f_s, 1)
        py /= 1000
                        
        return py.detach()
    
    
#     def optimize_var0(self, x_train, y_train, x_val, y_val, init_var0=100, lr=1):
#         logvar0 = torch.log(torch.tensor(init_var0).float())
#         logvar0.requires_grad = True
        
#         x_out = torch.from_numpy(np.random.uniform(-10, 10, size=[100, 2])).float()
#         y_out = torch.tensor([0.5]*100).float()

#         opt = torch.optim.Adam([logvar0], lr=lr)
#         pbar = trange(10, position=0, leave=True)
# #         pbar = range(10)

#         for _ in pbar:
#             var0 = logvar0.exp()
#             self.estimate_variance_batch(x_train, y_train, var0)

#             out_in = self.forward_linearized(x_val, progress=False)
#             loss = F.binary_cross_entropy_with_logits(out_in, y_val)
            
#             out_out = self.forward_linearized(x_out, progress=False)
#             loss += F.binary_cross_entropy_with_logits(out_out, y_out)

#             loss.backward()
#             opt.step()
#             opt.zero_grad()

#             pbar.set_description(f'var0: {logvar0.exp().item():.3f}, NLL: {loss.item():.3f}')

#         return logvar0.exp().detach()
    
    
    def optimize_var0(self, x_train, y_train, x_val, y_val, interval, rng_ood=(-3, 3)):
        var0s = interval
        nlls = []
        
        m, n = x_val.shape
        x_out = torch.from_numpy(np.random.uniform(*rng_ood, size=[m, n])).float()
        y_out = torch.tensor([0.5]*m).float()
        
        pbar = tqdm(var0s, position=0, leave=True)

        for var0 in pbar:
            self.estimate_variance_batch(x_train, y_train, var0)
            
            try:
                out_in = self.forward_linearized(x_val, progress=False)
                loss_in = F.binary_cross_entropy(out_in, y_val)

                out_out = self.forward_linearized(x_out, progress=False)
                loss_out = F.binary_cross_entropy(out_out, y_out)
#                 loss_out = -torch.mean(-out_out*torch.log(out_out + 1e-8) - (1-out_out)*torch.log(1-out_out + 1e-8))
                
                loss = loss_in + loss_out
                loss = np.nan_to_num(loss.detach().item(), nan=np.inf)
            except RuntimeError:
                # Error due to nan
                loss_in = np.inf
                loss_out = np.inf
                loss = np.inf
                
            nlls.append(loss)

            pbar.set_description(f'var0: {var0:.3f}, loss_in: {loss_in:.3f}, loss_out: {loss_out:.3f}, loss: {loss:.3f}')
            
        best_var0 = var0s[np.argmin(nlls)]

        return best_var0
    
    
    def optimize_var0_multi(self, x_train, y_train, x_val, y_val, interval, rng_ood=(-3, 3)):
        var0s = interval
        nlls = []
        
        m, n = x_val.shape
        x_out = torch.from_numpy(np.random.uniform(*rng_ood, size=[m, n])).float()
        y_out = torch.tensor([0.5]*m).float()
        
        pbar = tqdm(var0s, position=0, leave=True)

        for var0 in pbar:
            self.estimate_variance_batch(x_train, y_train, var0, multiclass=True)
            
            try:
                out_in = self.forward_linearized_multi(x_val, progress=False)
                loss_in = F.nll_loss(torch.log(out_in), y_val)

                out_out = self.forward_linearized_multi(x_out, progress=False)
                loss_out = torch.mean(torch.sum(-1/4 * torch.log(out_out), 1))
                                
                loss = loss_in + loss_out
                loss = np.nan_to_num(loss.detach().item(), nan=np.inf)
            except RuntimeError as e:
                print(str(e))
                # Error due to nan
                loss_in = np.inf
                loss_out = np.inf
                loss = np.inf
                
            nlls.append(loss)
            
            pbar.set_description(f'var0: {var0:.3f}, loss_in: {loss_in:.3f}, loss_out: {loss_out:.3f}, loss: {loss:.3f}')
            
        best_var0 = var0s[np.argmin(nlls)]

        return best_var0
        


def laplace_parameters(module, params):
#     mod_class = module.__class__.__name__
#     if mod_class not in ['Linear', 'Conv2d']:
#         return
    
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            # print(module, name)
            continue

        data = module._parameters[name].data
#         module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_var', data.new(data.size()).zero_())
#         module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


@torch.no_grad()
def predict(test_loader, model, n_samples=100):
    py = []

    # for x, y in tqdm(test_loader):
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1)

        py_ /= n_samples
        py.append(py_)

    return torch.cat(py, dim=0)


def vector_to_parameter_list(vec, parameters):
    """
    Convert the vector `vec` to a parameter-list format matching `parameters`.
    This function is the inverse of `parameters_to_vector` from the
    pytorch module `torch.nn.utils.convert_parameters`.
    Contrary to `vector_to_parameters`, which replaces the value
    of the parameters, this function leaves the parameters unchanged and
    returns a list of parameter views of the vector.
    ```
    from torch.nn.utils import parameters_to_vector
    vector_view = parameters_to_vector(parameters)
    param_list_view = vector_to_parameter_list(vec, parameters)
    for a, b in zip(parameters, param_list_view):
        assert torch.all_close(a, b)
    ```
    Parameters:
    -----------
        vec: Tensor
            a single vector represents the parameters of a model
        parameters: (Iterable[Tensor])
            an iterator of Tensors that are of the desired shapes.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'.format(
            torch.typename(vec)))
    params_new = []
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it
        param_new = vec[pointer:pointer + num_param].view_as(param).data
        params_new.append(param_new)
        # Increment the pointer
        pointer += num_param

    return params_new