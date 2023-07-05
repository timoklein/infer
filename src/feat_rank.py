import numpy as np
import torch
import torch.linalg as la

from .agent import InFerDDQN


@torch.inference_mode()
def calculate_feature_rank(
    agent: InFerDDQN,
    obs: torch.Tensor,
    epsilon: float,
) -> tuple[int, dict[str, float]]:
    """Approximate the feature rank of the current representation.

    Feature rank is defined as the number of singular values greater than some epsilon.
    See the paper for details: https://arxiv.org/pdf/2204.09560.pdf.

    """
    representation = agent.phi(obs)
    U, S, V = la.svd(representation)
    sigma_dict = {f"feat_rank/sigma_{k}": s for k, s in enumerate(S)}
    # NOTE: The sqrt scaling term (np.sqrt(obs.shape[0])) is not used to generate the plots in the paper AFAIK
    # If used, the feature rank would be significantly lower than the reported values in the paper
    # See page 20 of the paper:
    # "We then take the singular value decomposition of this matrix and count the number of singular values greater than 0.01
    # to get an estimate of the dimension of the network’s representation layer"
    return torch.sum(S > epsilon), sigma_dict
