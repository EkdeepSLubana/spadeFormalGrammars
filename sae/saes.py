import torch
from sparsemax import Sparsemax
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

class SAE(torch.nn.Module):
    def __init__(self, dimin=2, width=5, sae_type='relu', kval_topk=None, mp_kval=None,
                 normalize_weights=False, lambda_init = None):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'jumprelu', 'sparsemax_lintx', 'sparsemax_dist'
        kval_topk: (int)
            k in topk sae_type
        normalize_weights: (bool)
            whether to normalize the decoder weights to unit norm
        """
        super(SAE, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.eps = 0.01
        self.normalize_weights = normalize_weights

        ## Encoder parameters
        self.Ae = nn.Parameter(torch.randn((width, dimin))) #N(0,1) init
        if not 'MP' in sae_type:
            self.be = nn.Parameter(torch.zeros((1, width)))

        ## Decoder parameters
        self.bd = nn.Parameter(torch.zeros((1, dimin)))
        if not 'MP' in sae_type:
            self.Ad = nn.Parameter(torch.randn((dimin, width))) #N(0,1) init
            with torch.no_grad():
                self.Ad.copy_(self.Ae.T) #at init, decoder is the transpose of encoder

        ## Normalize the encoder and decoder weights, if desired
        if normalize_weights:
            print('\n\n==== Normalizing encoder / decoder weights ====\n\n')
            with torch.no_grad():
                Ae_unit = self.Ae / (self.eps + torch.linalg.norm(self.Ae, dim=1, keepdim=True))
                self.Ae.copy_(Ae_unit)
                if not 'MP' in sae_type:
                    Ad_unit = self.Ad / (self.eps + torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                    self.Ad.copy_(Ad_unit)

        ## Parameters for specific SAEs
        # JumpReLU
        if sae_type=='jumprelu':
            self.logthreshold = nn.Parameter(torch.log(1e-3*torch.ones((1, width))))
            self.bandwidth = 1e-3 #width of rectangle used in approx grad of jumprelu wrt threshold

        # Sparsemax
        if 'sparsemax' in sae_type:
            lambda_init = 1/(width*dimin) if lambda_init is None else lambda_init
            lambda_pre = softplus_inverse(lambda_init)
            self.lambda_pre = nn.Parameter(lambda_pre) #trainable parameter (~inv temp) for sparsemax
        else:
            lambda_init = 1/(4*dimin) if lambda_init is None else lambda_init
            lambda_pre = softplus_inverse(lambda_init)
            self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False) #not trainable
        
        # Topk parameter
        if sae_type=='topk':
            if kval_topk is not None:
                self.kval_topk = kval_topk
            else:
                raise ValueError('kval_topk must be provided for topk sae_type')

        # MP parameter
        if sae_type=='MP':
            if mp_kval is not None:
                self.mp_kval = mp_kval
            else:
                raise ValueError('kval_topk must be provided for topk sae_type')


    @property
    def lambda_val(self): #lambda_val is lambda, forced to be positive here
        return F.softplus(self.lambda_pre)


    def forward(self, x, return_hidden=False, inf_k=None):
        lam = self.lambda_val

        ## Vanilla
        if self.sae_type=='relu':
            # Encoder projection
            x = x-self.bd 
            x = torch.matmul(x, self.Ae.T) + self.be

            # Sparse code: ReLU
            codes = F.relu(lam*x)

            # Reconstruction
            x = torch.matmul(codes, self.Ad.T) + self.bd

        ## TopK
        elif self.sae_type=='topk':
            kval = self.kval_topk if inf_k is None else inf_k
            
            # Encoder projection
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T)

            # Sparse code: Top-K selection
            _, topk_indices = torch.topk(F.relu(x), kval, dim=-1)
            mask = torch.zeros_like(x) 
            mask.scatter_(-1, topk_indices, 1)

            # Codes
            codes = x * mask * lam

            # Reconstruction
            x = torch.matmul(codes, self.Ad.T) + self.bd

        ## Matching Pursuits
        elif self.sae_type=='MP':
            kval = self.mp_kval if inf_k is None else inf_k
            x = x - self.bd # Pre-encoder bias

            # Initialize codes
            codes = torch.zeros(x.shape[0], self.Ae.shape[0], device=x.device)

            # Sparse code: Greedy selection of dictionary atoms
            for _ in range(kval):
                # Encoder projection
                z = x @ self.Ae.T  
                val, idx = torch.max(z, dim=1)

                # Add top concept to the current codes
                to_add = torch.nn.functional.one_hot(idx, num_classes=codes.shape[1]).float()
                to_add = to_add * val.unsqueeze(1)
                to_add = to_add * lam

                # Accumulate contribution and update residual
                codes = codes + to_add
                x = x - to_add @ self.Ae

            # Reconstruction
            x = torch.matmul(codes, self.Ae) + self.bd

        ## JumpReLU
        elif self.sae_type=='jumprelu':
            # Encoder projection
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T) + self.be

            # Sparse code: Heavystep + ReLU function
            x = F.relu(lam*x)
            threshold = torch.exp(self.logthreshold)
            codes = jumprelu(x, threshold, self.bandwidth)

            # Reconstruction
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type=='sparsemax_dist':
            # Encoder projection
            x = -lam*torch.square(torch.norm(x.unsqueeze(1)-self.Ae.unsqueeze(0), dim=-1))

            # Sparse code: Sparsemax computation
            sm = Sparsemax(dim=-1)
            codes = sm(x)

            # Reconstruction
            x = torch.matmul(codes, self.Ad.T)

        else:
            raise ValueError('Invalid sae_type')

        if not return_hidden:
            return x
        else:
            return x, codes
        
        
    def decode(self, codes):
        if self.sae_type == 'relu':
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == 'topk':
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == 'MP':
            x = torch.matmul(codes, self.Ae) + self.bd
        elif self.sae_type == 'jumprelu':
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == 'sparsemax_dist':
            x = torch.matmul(codes, self.Ad.T)
        
        return x

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.Ae)
        init.zeros_(self.Ad)
        init.zeros_(self.be)
        init.zeros_(self.bd)

        # decoder weights are initialized to random unit vectors
        # dec_weight = torch.randn_like(self.Ad.weight)
        # dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        # self.Ad.weight = nn.Parameter(dec_weight)
        # self.Ae.weight = nn.Parameter(dec_weight.clone().T)


######################## UTILS ########################
def rectangle(x):
    # rectangle function
    return ((x >= -0.5) & (x <= 0.5)).float()

class JumpReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device="cuda")
        ctx.save_for_backward(x, threshold, bandwidth)
        return x*(x>threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors

        # Compute gradients
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)  # Aggregating across batch dimension
        
        return x_grad, threshold_grad, None  # None for bandwidth since const

def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)



def softplus_inverse(input, beta=1.0, threshold=20.0):
        """"
        inverse of the softplus function in torch
        """
        if isinstance(input, float):
                input = torch.tensor([input])
        if input*beta<threshold:
                return (1/beta)*torch.log(torch.exp(beta*input)-1.0)
        else:
              return input[0]


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=input.dtype, device=input.device)
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0*grad_output #no ste to input
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None  # None for bandwidth since const

def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)