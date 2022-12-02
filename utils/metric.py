import torch


class Evaluator():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def Average_Accuracy(self):
        return {'top1': self.top1.Average_Accuracy(), 'top5': self.top5.Average_Accuracy()}

    def reset(self):
        self.top1.reset()
        self.top5.reset()

    def update(self, acc, size):
        self.top1.update(acc[0], size)
        self.top5.update(acc[1], size)

    def sync(self, device):
        self.top1.sync(device)
        self.top5.sync(device)


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def Average_Accuracy(self):
        return self.sum / self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def sync(self, device):
        _val = torch.Tensor([self.val]).to(device)
        _sum = torch.Tensor([self.sum]).to(device)
        _count = torch.Tensor([self.count]).to(device)

        torch.distributed.reduce(_val, dst=0)
        torch.distributed.reduce(_sum, dst=0)
        torch.distributed.reduce(_count, dst=0)

        if torch.distributed.get_rank() == 0:
            self.val = _val.item()
            self.sum = _sum.item()
            self.count = _count.item()
