import torch
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random

from .AL import AL


class PCB(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device 
        self.pred = []
        self.statistics = statistics 

    def select(self, n_query, **kwargs):
        self.pred = []
        self.model.eval()
        # embDim = self.model.image_encoder.attnpool.c_proj.out_features
        num_unlabeled = len(self.U_index)
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            
            # generate entire unlabeled features set
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)
                out, features = self.model(inputs, get_feature=True)
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)
                # _, preds = torch.max(out.data, 1)
                self.pred.append(maxInds.detach().cpu())
        self.pred = torch.cat(self.pred)
        
        Q_index = []
        
        while len(Q_index) < n_query:
            min_cls = int(torch.argmin(self.statistics))
            sub_pred = (self.pred == min_cls).nonzero().squeeze(dim=1).tolist()
            if len(sub_pred) == 0:
                num = random.randint(0, num_unlabeled-1)
                while num in Q_index:
                    num = random.randint(0, num_unlabeled-1)
                Q_index.append(num)
            else:
                random.shuffle(sub_pred)
                for idx in sub_pred:
                    if idx not in Q_index:
                        Q_index.append(idx)
                        self.statistics[min_cls] += 1
                        break 
                else: 
                    num = random.randint(0, num_unlabeled-1)
                    while num in Q_index:
                        num = random.randint(0, num_unlabeled-1)
                    Q_index.append(num)
            
        Q_index = [self.U_index[idx] for idx in Q_index]
        
        return Q_index