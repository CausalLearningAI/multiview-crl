"""Definition of loss functions."""

from abc import ABC, abstractmethod

import numpy as np
import torch


# for numerical experiment
class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z_rec, z3_rec, l):
        """
        z1_t = h(z1)
        z2_t = h(z2)
        z3_t = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        and z2 ~ p(z2 | z1)

        returns the total loss and componentwise contributions
        """

    def __call__(self, z_rec, z3_rec, l):
        return self.loss(z_rec, z3_rec, l)


class LpSimCLRLoss(CLLoss):
    """Extended InfoNCE objective for non-normalized representations based on an Lp norm.

    Args:
        p: Exponent of the norm to use.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        simclr_compatibility_mode: Use logsumexp (as used in SimCLR loss) instead of logmeanexp
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
        self,
        p: int = 2,
        tau: float = 1.0,
        alpha: float = 0.5,
        simclr_compatibility_mode: bool = False,
        simclr_denominator: bool = True,
        pow: bool = True,
    ):
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.simclr_compatibility_mode = simclr_compatibility_mode
        self.simclr_denominator = simclr_denominator
        self.pow = pow

    def loss(self, z_rec, z3_rec, l):
        """
        Calculates the loss function for the given inputs.

        Args:
            z_rec (list): List of reconstructed z values.
            z3_rec (list): List of reconstructed z3 values.
            l (int): Length of the input lists.

        Returns:
            tuple: A tuple containing the mean loss, the loss array, and a list of mean positive and negative losses.
        """
        # del z1, z2_con_z1, z3
        neg = 0
        pos = 0
        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            for i in range(l):
                neg = neg + torch.norm(
                    torch.abs(z_rec[i].unsqueeze(0) - z3_rec[i].unsqueeze(1) + 1e-12),
                    p=self.p,
                    dim=-1,
                )
            for i in range(l - 1):
                pos = torch.norm(torch.abs(z_rec[i] - z_rec[i + 1]) + 1e-12, p=self.p, dim=-1)
        else:
            for i in range(l):
                neg = neg + torch.pow(z_rec[i].unsqueeze(1) - z3_rec[i].unsqueeze(0), float(self.p)).sum(dim=-1)
            for i in range(l - 1):
                pos = pos + torch.pow(z_rec[i] - z_rec[i + 1], float(self.p)).sum(dim=-1)

        if not self.pow:
            neg = neg.pow(1.0 / self.p)
            pos = pos.pow(1.0 / self.p)

        if self.simclr_compatibility_mode:
            neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

            loss_pos = pos / self.tau
            loss_neg = torch.logsumexp(-neg_and_pos / self.tau, dim=1)
        else:
            if self.simclr_denominator:
                neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)
            else:
                neg_and_pos = neg

            loss_pos = pos / self.tau
            loss_neg = _logmeanexp(-neg_and_pos / self.tau, dim=1)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)
        # loss_std = torch.std(loss)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


def _logmeanexp(x, dim):
    """
    Compute the log-mean-exponential of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the log-mean-exponential.

    Returns:
        torch.Tensor: The log-mean-exponential of the input tensor along the specified dimension.
    """
    N = torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    return torch.logsumexp(x, dim=dim) - torch.log(N)


class UnifiedCLLoss(CLLoss):
    """Loss for view-specific encoders"""

    def __init__(
        self,
        base_loss: CLLoss,
    ):
        """
        Initializes the UnifiedCLLoss class.

        Args:
            base_loss (CLLoss): The base loss function.

        """
        self.base_loss = base_loss

    def loss(self, est_content_dict: dict, z_rec, z3_rec):
        """
        Computes the loss for all subsets of views.

        Args:
            est_content_dict (dict): A dictionary containing the estimated content indices for each subset.
            z_rec: The reconstructed z values.
            z3_rec: The reconstructed z3 values.

        Returns:
            tuple: A tuple containing the total loss mean, total loss,
            and a list of total loss means for positive and negative samples.

        """
        z_rec = torch.stack(z_rec, dim=0)  # [n_views, batch-size, nSk]
        z3_rec = torch.stack(z3_rec, dim=0)  # [n_views, batch-size, nSk]

        total_loss_mean, total_loss, total_loss_pos_mean, total_loss_neg_mean = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for subset, subset_c_ind in est_content_dict.items():
            l = len(subset)
            c_ind = (
                torch.from_numpy(np.stack(list(subset_c_ind.values()))).type_as(z_rec).long()
            )  # n_views_in_this_subset, batch_size, n_Sk
            c_ind = c_ind[:, None, :].repeat(1, z_rec.shape[1], 1)  # (n_views_in_this_subset, batch_size, content_size)

            hz = torch.gather(z_rec[list(subset), :], -1, c_ind)
            hz3 = torch.gather(z3_rec[list(subset), :], -1, c_ind)
            loss_mean, loss, loss_mean_list = self.base_loss.loss(z_rec=hz, z3_rec=hz3, l=l)
            total_loss_mean += loss_mean
            total_loss += loss
            total_loss_pos_mean += loss_mean_list[0]
            total_loss_neg_mean += loss_mean_list[1]
        return total_loss_mean, total_loss, [total_loss_pos_mean, total_loss_neg_mean]


# for multimodal experiment
def infonce_loss(hz, sim_metric, criterion, projector=None, tau=1.0, estimated_content_indices=None, subsets=None):
    """
    Calculates the sum of InfoNCE loss for a given input tensor `hz`, over all subsets.

    Args:
        hz (torch.Tensor): The input tensor of shape (batch_size, ..., num_features).
        sim_metric: The similarity metric used for calculating the loss.
        criterion: The loss criterion used for calculating the loss.
        projector: The projector used for projecting the input tensor (optional).
        tau (float): The temperature parameter for the loss calculation (default: 1.0).
        estimated_content_indices: The estimated content indices (optional).
        subsets: The subsets of indices used for calculating the loss (optional).

    Returns:
        torch.Tensor: The calculated InfoNCE loss.

    """
    if estimated_content_indices is None:
        return infonce_base_loss(hz, sim_metric, criterion, projector, tau)
    else:
        total_loss = torch.zeros(1).type_as(hz)
        for est_content_indices, subset in zip(estimated_content_indices, subsets):
            total_loss += infonce_base_loss(
                hz[list(subset), ...], est_content_indices, sim_metric, criterion, projector, tau
            )
        return total_loss


def infonce_base_loss(hz_subset, content_indices, sim_metric, criterion, projector=None, tau=1.0):
    """
    Computes the InfoNCE (Normalized Cross Entropy) loss for multi-view data.

    Args:
        hz_subset (list): List of tensors representing the latent space of each view.
        content_indices (list): List of indices representing the content dimensions.
        sim_metric (function): Similarity metric function to compute pairwise similarities.
        criterion (function): Loss criterion function.
        projector (function, optional): Projection function to project the latent space. Defaults to None.
        tau (float, optional): Temperature parameter for similarity computation. Defaults to 1.0.

    Returns:
        torch.Tensor: Total loss value.

    """

    n_view = len(hz_subset)
    SIM = [
        [torch.Tensor().type_as(hz_subset) for _ in range(n_view)] for _ in range(n_view)
    ]  # n_views x n_view x batch_size (d) x batch_size (d)

    projector = projector or (lambda x: x)

    for i in range(n_view):
        for j in range(n_view):
            if j >= i:
                # compute similarity matrix using projected latents
                sim_ij = (
                    sim_metric(  # (hz[i]: n_views, n_latent_dim)
                        projector(hz_subset[i].unsqueeze(-2)),
                        projector(
                            hz_subset[j].unsqueeze(-3)
                        ),  # (bs, 1, n_latent_dim) and (1, bs n_latent_dim) -> bs , bs
                    )
                    / tau
                ).type_as(hz_subset)
                # compute positive pairs using (diagonal elements) only the content dimensions
                pos_sim_ij = (
                    sim_metric(  # (hz[i]: n_views, n_latent_dim)
                        hz_subset[i].unsqueeze(-2)[..., content_indices],
                        hz_subset[j].unsqueeze(-3)[
                            ..., content_indices
                        ],  # (bs, 1, n_latent_dim) and (1, bs n_latent_dim) -> bs , bs
                    )
                    / tau
                ).type_as(hz_subset)
                sim_ij = pos_sim_ij
                if i == j:
                    d = sim_ij.shape[-1]  # batch size
                    sim_ij[..., range(d), range(d)] = float("-inf")
                SIM[i][j] = sim_ij
            else:
                SIM[i][j] = SIM[j][i].transpose(-1, -2).type_as(hz_subset)

    total_loss_value = torch.zeros(1).type_as(hz_subset)
    for i in range(n_view):
        for j in range(n_view):
            if i < j:
                raw_scores = []
                raw_scores1 = torch.cat([SIM[i][j], SIM[i][i]], dim=-1).type_as(hz_subset)
                raw_scores2 = torch.cat([SIM[j][j], SIM[j][i]], dim=-1).type_as(hz_subset)
                raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)  # d, 2d
                targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
                total_loss_value += criterion(raw_scores, targets)
    return total_loss_value
