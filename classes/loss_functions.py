import torch


class LossCompute:
    """A simple loss compute and train function."""

    def __init__(self, model, criterion_tlog, opt=None, ):
        self.model = model
        self.criterion_tlog = criterion_tlog
        self.opt = opt

    def __call__(self, out, y, y_mask, norm=1.0, is_test=False):


        dist = torch.sum((out[:, 0, :] - self.model.c) ** 2, dim=1)
        loss_tlog = torch.mean((1 - y) * torch.sqrt(dist) - y * torch.log(1 - torch.exp(-torch.sqrt(dist))))



        if not is_test:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item(),loss_tlog
        # return loss.item() * norm
