# https://github.com/Naeem-Paeedeh/ADAPTER
# @article{paeedeh2024cross,
#   title={Cross-domain few-shot learning via adaptive transformer networks},
#   author={Paeedeh, Naeem and Pratama, Mahardhika and Ma’sum, Muhammad Anwar and Mayer, Wolfgang and Cao, Zehong and Kowlczyk, Ryszard},
#   journal={Knowledge-Based Systems},
#   volume={288},
#   pages={111458},
#   year={2024},
#   publisher={Elsevier}
# }

# @article{Paeedeh2020ImprovingTB,
#   title={Improving the Backpropagation Algorithm with Consequentialism Weight Updates over Mini-Batches},
#   author={Naeem Paeedeh and Kamaledin Ghiasi-Shirazi},
#   journal={Neurocomputing},
#   year={2020},
#   volume={461},
#   pages={86-98},
#   url={https://api.semanticscholar.org/CorpusID:212657435}
# }


# @inproceedings{Zhou2003LearningWL,
#   title={Learning with Local and Global Consistency},
#   author={Dengyong Zhou and Olivier Bousquet and Thomas Navin Lal and Jason Weston and Bernhard Scholkopf},
#   booktitle={Neural Information Processing Systems},
#   year={2003},
#   url={https://api.semanticscholar.org/CorpusID:508435}
# }
# This code is modified from https://github.com/provezano/lgc
import torch
from torch import Tensor as T
import shared as sh
import new_types as nt


def pinv_robust(x, coef, device):
    dim1 = x.size()[0]
    dim2 = x.size()[1]
    x = x.to(device)

    if dim1 >= dim2:
        cor = x.t() @ x
        idn = coef * torch.eye(dim2, dim2).to(device)
        nv = (cor + idn).inverse() @ x.t()
    else:
        cor = x @ x.t()
        idn = coef * torch.eye(dim1, dim1).to(device)
        nv = x.t() @ (cor + idn).inverse()
    return nv


class LGC:
    """
    Learning with Local and Global Consistency (LGC) algorithm
    """
    
    def __init__(self,
                 alpha=0.9,
                 sigma=50,
                 rcond_for_pinv=1e-5,
                 distance_function: nt.DistanceFunction = nt.DistanceFunction.Euclidean,
                 ):
        self.alpha = alpha
        self.sigma = sigma
        self.rcond_for_pinv = rcond_for_pinv
        self.distance_function = distance_function

    def _calculate_A_hat(self, A):
        diag_vec = A.sum(dim=1)
        D_sqrt_inv = torch.diag(1.0 / diag_vec.sqrt())
        return D_sqrt_inv @ A @ D_sqrt_inv

    def compute(self, x: T, y_bar: T):
        # Regarding the labeling, the first part of the Y_bar must be the labeled part of the target domain.
        # The rest is dedicated to the unlabeled samples. Therefore, the first part of Y_bar is one-hot encoded,
        # but the rest elements of the remaining parts for unlabeled samples must be zeros.
        
        assert x.shape[0] == y_bar.shape[0]

        device = 'cpu'  # device = x.device
        
        # if self.identity is None or y_bar.shape[0] > self.identity.shape[0]:
        identity = torch.eye(y_bar.shape[0], device=device)
            
        x, y_bar = sh.to_device([x, y_bar], device=device)

        if self.distance_function == nt.DistanceFunction.Euclidean:
            # "cdist computes batched the p-norm distance between each pair of the two collections of row vectors"
            distance_matrix = torch.cdist(x, x)
        elif self.distance_function == nt.DistanceFunction.CosineDistance:
            sim, _ = sh.cosine_similarity(x, x)  # Between -1, 1
            distance_matrix = 2.0 * (1.0 - sim)  # (1.0 - sim) is between 0, 2
        elif self.distance_function == nt.DistanceFunction.CosineSimilarity:
            sim, _ = sh.cosine_similarity(x, x)
            distance_matrix = 2.0 * (1.0 - sim)
            # distance_matrix = F.softmax(distance_matrix, dim=-1)
        else:
            raise NotImplementedError
        
        # A is the affinity matrix
        A = torch.exp(-(distance_matrix * distance_matrix) / (2.0 * (self.sigma ** 2)))   # RBF
        A.fill_diagonal_(0)
        A_hat = self._calculate_A_hat(A)
        
        # f_star = (self.identity - self.alpha * A_hat).cpu().pinverse(self.rcond_for_pinv).to(device) @ y_bar
        f_star = pinv_robust(identity - self.alpha * A_hat, self.rcond_for_pinv, device) @ y_bar

        return f_star
