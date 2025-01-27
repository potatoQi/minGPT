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
    
    def __init__(self, cfg, model, optimizer, train_data, test_data):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True)

        if cfg.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = cfg.device
        self.model = self.model.to(self.device)

    def run_eval(self, len):
        self.eval_loop(self.model, self.test_dataloader, len)

    def eval_loop(self, model, dataloader, length):
        model.eval()
        device = self.device

        samples_size = len(dataloader.dataset)
        acc_sum = 0

        for batch, (X, y) in enumerate(dataloader):
            '''
            原数组有 length 个数, [x, x, x, x, x, x]
            经过排序后, 将两个结果拼接在一起: [x', x, x, x, x, x, y, y, y, y, y, y']
            去掉最后一个数, 就是 X: [x', x, x, x, x, x, y, y, y, y, y]
            去掉第一个数, 然后把前 length 个数变为 -1, 就是 y
            '''
            X = X[:, :length].to(device)
            y = torch.sort(X, dim=1)[0].to(device)

            pred = model.generate(X, length, do_sample=False)
            pred = pred[:, length:]

            acc_sum += (pred == y).all(dim=1).sum()
        
        print(f'acc: {(acc_sum/samples_size):.2f}')

    def run_train(self):
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