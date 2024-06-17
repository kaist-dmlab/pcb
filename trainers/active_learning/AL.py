import torch

class AL(object):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, **kwargs):
        self.unlabeled_dst = unlabeled_dst
        self.U_index = U_index
        self.unlabeled_set = torch.utils.data.Subset(unlabeled_dst, U_index)
        self.n_unlabeled = len(self.unlabeled_set)
        self.n_class = n_class
        self.model = model
        self.index = []
        self.cfg = cfg

    def select(self, **kwargs):
        return