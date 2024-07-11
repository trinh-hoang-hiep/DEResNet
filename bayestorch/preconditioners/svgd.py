# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Stein variational gradient descent preconditioner."""

from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
#######################################################gpm
import numpy as np
import xarray as xr
if torch.cuda.is_available():#############################
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

__all__ = [
    "SVGD",
]


class SVGD(Optimizer):
    """Stein variational gradient descent preconditioner.

    References
    ----------
    .. [1] Q. Liu and D. Wang.
           "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm".
           In: Advances in Neural Information Processing Systems. 2016, pp. 2378-2386.
           URL: https://arxiv.org/abs/1608.04471

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>>
    >>> from bayestorch.kernels import RBFSteinKernel
    >>> from bayestorch.preconditioners import SVGD
    >>>
    >>>
    >>> num_particles = 5
    >>> batch_size = 10
    >>> in_features = 4
    >>> out_features = 2
    >>> models = nn.ModuleList(
    ...     [nn.Linear(in_features, out_features) for _ in range(num_particles)]
    ... )
    >>> kernel = RBFSteinKernel()
    >>> preconditioner = SVGD(models.parameters(), kernel, num_particles)
    >>> input = torch.rand(batch_size, in_features)
    >>> outputs = torch.cat([model(input) for model in models])
    >>> loss = outputs.sum()
    >>> loss.backward()
    >>> preconditioner.step()

    """

    # override
    def __init__(
        self,
        params: "Union[Iterable[Tensor], Iterable[Dict[str, Any]]]",
        kernel: "Callable[[Tensor], Tuple[Tensor, Tensor]]",
        num_particles: "int",
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to precondition. The total number of
            parameters must be a multiple of `num_particles`.
        kernel:
            The kernel, i.e. a callable that receives as an argument the
            particles and returns the corresponding kernels and kernel
            gradients.
        num_particles:
            The number of particles.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        super().__init__(params, {"kernel": kernel, "num_particles": num_particles})
        self.kernel = kernel
        self.num_particles = num_particles = int(num_particles)

        # Check consistency between number of parameters
        # and number of particles for each group
        for group in self.param_groups:
            params = group["params"]
            num_particles = group["num_particles"]

            if num_particles < 1 or not float(num_particles).is_integer():
                raise ValueError(
                    f"`num_particles` ({num_particles}) must be in the integer interval [1, inf)"
                )

            # Extract particles
            particles = nn.utils.parameters_to_vector(params)

            if particles.numel() % num_particles != 0:
                raise ValueError(
                    f"Total number of parameters ({particles.numel()}) must "
                    f"be a multiple of `num_particles` ({num_particles})"
                )

    # override
    @torch.no_grad()
    def step(self) -> "None":
        for group in self.param_groups:
            params = group["params"]
            kernel = group["kernel"]
            num_particles = group["num_particles"]

            # Extract particles
            particles = nn.utils.parameters_to_vector(params).reshape(num_particles, -1)

            # Extract particle gradients
            particle_grads = []
            for param in params:
                grad = param.grad
                if grad is None:
                    raise RuntimeError("Gradient of some parameters is None")
                particle_grads.append(grad)
            particle_grads = nn.utils.parameters_to_vector(particle_grads).reshape(
                num_particles, -1
            )

            # Compute kernels and kernel gradients
            kernels, kernel_grads = kernel(particles)

            # Attractive gradients (already divided by `num_particles` => use `NLUPLoss` with reduction="mean")
            particle_grads = kernels @ particle_grads
            cos_1 = torch.nn.CosineSimilarity(dim=0)
            # # a=cos_1(kernel_grads, particle_grads)
            # # Repulsive gradients
            # particle_grads -= (kernel_grads-kernel_grads*cos_1(kernel_grads, particle_grads)) / num_particles########################################gpm: sd svd, pca, update mà ko thay đổi input 
            # # # Repulsive gradients
            # # particle_grads -= kernel_grads / num_particles

            # Flatten
            particle_grads = particle_grads.flatten()

            # # Inject particle gradients
            # start_idx = 0
            # for param in params:
            #     num_elements = param.numel()
            #     param.grad = particle_grads[
            #         start_idx : start_idx + num_elements
            #     ].reshape_as(param)
            #     start_idx += num_elements

            kernel_grads = kernel_grads.flatten()##########################thêm svgd sin 
            # Inject particle gradients
            start_idx = 0
            for param in params:
                num_elements = param.numel()
                particle_gradstmp = particle_grads[
                    start_idx : start_idx + num_elements
                ].reshape_as(param)
                kernel_gradstmp = kernel_grads[
                    start_idx : start_idx + num_elements
                ].reshape_as(param)
                # # particle_gradstmp-=(kernel_gradstmp-kernel_gradstmp*cos_1(kernel_gradstmp, particle_gradstmp)) / num_particles
                # particle_gradstmp=(particle_gradstmp-(kernel_gradstmp-kernel_gradstmp*cos_1(kernel_gradstmp, particle_gradstmp)) / (num_particles))##/10.0######cài đặ cũ ở trên ################## 
                particle_gradstmp=(particle_gradstmp-kernel_gradstmp / (num_particles))

#                 # particle_gradstmp=param.grad +(particle_gradstmp-(kernel_gradstmp-kernel_gradstmp*cos_1(kernel_gradstmp, particle_gradstmp)) / (num_particles))###/10.0########################optim3 
#                 if (len(particle_gradstmp.size())<2): 
#                     particle_gradstmp=(particle_gradstmp-(kernel_gradstmp-kernel_gradstmp*cos_1(kernel_gradstmp, particle_gradstmp)) / (num_particles))###/10.0########################optim3 

#                 else:
#                     feature_list=GP_SVD(particle_gradstmp)
#                     if(len(feature_list.shape)==2):
#                         feature_mat=torch.Tensor(np.dot(feature_list,feature_list.transpose())).to(device)
#                         sz =  kernel_gradstmp.size(0)
#                         particle_gradstmp=particle_gradstmp -(kernel_gradstmp-torch.mm(kernel_gradstmp.view(sz,-1),feature_mat).view(particle_gradstmp.size()))/(num_particles)##################################gpm 
#                     elif( feature_list.size == 0):
#                         particle_gradstmp=particle_gradstmp -(kernel_gradstmp)/(num_particles)##################################gpm 
#                     else:
# # #                         feature_mat=torch.Tensor(np.einsum('ijkl,nolp->ijlp',feature_list,feature_list.transpose())).to(device)###########https://stackoverflow.com/questions/71004414/numpy-dot-for-dimensions-2
# #                         xA   = xr.DataArray(feature_list, dims=['d1','d2','d3','d4'])
# #                         xA_t = xA.T #################################https://stackoverflow.com/questions/52679673/numpy-dot-product-of-a-4d-array-with-its-transpose-fails
# #                         feature_mat=xr.dot(xA,xA_t, dims=['d1','d2']).to_numpy()
# #                         feature_mat=torch.Tensor(feature_mat).to(device)
#                         #######reshape last
#                         feature_list = feature_list.reshape(-1, feature_list.shape[-1])  ####https://stackoverflow.com/questions/18757742/how-to-flatten-only-some-dimensions-of-a-numpy-array
#                         feature_mat=torch.Tensor(np.dot(feature_list,feature_list.transpose())).to(device)
#                         sz =  kernel_gradstmp.size(0)
#                         particle_gradstmp=particle_gradstmp -(kernel_gradstmp-torch.mm(kernel_gradstmp.view(sz,-1),feature_mat).view(particle_gradstmp.size()))/(num_particles)##################################gpm 

#                     particle_gradstmp=particle_gradstmp
                param.grad=particle_gradstmp
                start_idx += num_elements
# def GP_SVD (activation):
#     act=activation.detach().cpu().numpy()
#     U,S,Vh = np.linalg.svd(act.transpose(), full_matrices=False)
#     # criteria (Eq-5)
#     sval_total = (S**2).sum()
#     sval_ratio = (S**2)/sval_total
#     r = np.sum(np.cumsum(sval_ratio)<0.95) #+1  
#     return U[:,0:r]