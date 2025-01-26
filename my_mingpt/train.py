from my_mingpt.utils import CfgNode
import torch
from torch.utils.data import DataLoader

class Trainer:
    ''' 训练类, 通用的一个类 '''

    @staticmethod
    def get_default_config():
        cfg = CfgNode()
        cfg.device = 'auto'
        cfg.Epochs = None
        cfg.batch_size = 64
        cfg.grad_norm_clip = 1.0
        return cfg
    
    def __init__(self, cfg, model, optimizer, train_data):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        
        if cfg.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = cfg.device
        self.model = self.model.to(self.device)

    def run(self):
        for t in range(1, self.cfg.Epochs + 1):
            print(f'Epoch: {t} --------------------------------\n')
            self.train_loop(self.model, self.optimizer, self.train_dataloader)

    def train_loop(self, model, optimizer, dataloader):
        model.train()
        device = self.device
        grad_norm_clip = self.cfg.grad_norm_clip
        batch_size = self.cfg.batch_size

        samples_size = len(dataloader.dataset)
        batches_size = len(dataloader)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits, loss = model(X, y)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.zero_grad()

            if batch % 100 == 0 or batch == batches_size - 1:
                print(f'loss:{loss:>7.2f} | {batch * batch_size + len(y)}/{samples_size}')