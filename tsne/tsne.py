from sklearn import metrics
import numpy as np
import sys
import scipy
from .utils import binary_search
import torch


class TSNE:
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
    
    @staticmethod
    def _calculate_perplexity(probs: np.ndarray) -> float:
        probs = probs[np.logical_not(np.isclose(probs, 0))]
        return 2 ** (-np.sum(probs * np.log2(probs)))
    
    @staticmethod
    def _calculate_conditional_probs(
        square_distances: np.ndarray,
        ix: int,
        sigma: float
    ) -> np.ndarray:
        numenator = np.exp(
            -square_distances[ix] / (2 * (sigma ** 2))
        )
        numenator[ix] = 0
        return numenator / numenator.sum()
    
    @staticmethod
    def _calculate_joint_probs_q(
        Y: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pairwise_distances = torch.cdist(Y, Y)
        square_distances = pairwise_distances.pow(2)

        qs = (1 + square_distances).pow(-1)
        ind = np.diag_indices(qs.shape[0])
        qs[ind[0], ind[1]] *= 0
        ind = np.triu_indices(n=qs.shape[0])
        qs = qs[ind]
        qs = qs / qs.sum()

        return qs
    
    @staticmethod
    def _calculate_kl_distance(
        P: torch.FloatTensor,
        Q: torch.FloatTensor
    ) -> torch.FloatTensor:
        p_zeros_mask = torch.isclose(P, torch.FloatTensor([0.0]))
        P, Q = P[~p_zeros_mask], Q[~p_zeros_mask]
        kl_distance = (P * torch.log(P / Q)).sum()
        return kl_distance

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        square_distances = np.power(metrics.pairwise_distances(X, X), 2)
        
        sigmas = [
            binary_search(
                func=lambda sigma: TSNE._calculate_perplexity(
                    TSNE._calculate_conditional_probs(
                        square_distances,
                        ix=ix,
                        sigma=sigma)
                ),
                target=self.perplexity
            )
            for ix in range(len(X))
        ]

        conditional_probs = [
            TSNE._calculate_conditional_probs(
                square_distances,
                ix=ix,
                sigma=sigma
            )
            for ix, sigma in enumerate(sigmas)
        ]
        conditional_probs = np.asarray(conditional_probs)

        target_joint_probs = conditional_probs[
            np.triu_indices(n=conditional_probs.shape[0])
        ]

        target_joint_probs = torch.tensor(
            target_joint_probs, dtype=torch.float32)
        
        Y = torch.randn(X.shape[0], self.n_components,
            requires_grad=True, dtype=torch.float32)

        opt = torch.optim.SGD(
            params=[Y],
            lr=1.0,
            momentum=0.8,
            nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        for iter_ in range(self.n_iter):
            print(f"Iter [{iter_} / {self.n_iter}]")
            joint_probs_q = TSNE._calculate_joint_probs_q(Y)
            kl_distance = TSNE._calculate_kl_distance(
                target_joint_probs,
                joint_probs_q
            )
            print(f"KL distance: {kl_distance.cpu().detach().item()}")

            opt.zero_grad()
            kl_distance.backward()
            opt.step()
            scheduler.step(kl_distance)
        
        return Y.detach().numpy()
