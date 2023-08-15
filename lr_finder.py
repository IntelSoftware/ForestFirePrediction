import torch
import intel_extension_for_pytorch
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
#import wandb


class LearningRateFinder:
    def __init__(
        self,
        model,
        optimizer,
        device,
        loss_fn=torch.nn.CrossEntropyLoss(),
        precision="fp32",
        #use_wandb=False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.precision = precision
        #self.use_wandb = use_wandb

    def forward_pass(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        total = labels.numel()
        return loss, correct, total

    def lr_range_test(self, train_dataloader, start_lr=1e-7, end_lr=1e-2, num_iter=100):
        orig_model_state_dict = self.model.state_dict()
        orig_opt_state_dict = self.optimizer.state_dict()
        lrs = []
        losses = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr
        lr_lambda = lambda x: (end_lr / start_lr) ** (x / num_iter)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            if i >= num_iter:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.precision == "bf16":
                with getattr(torch, f"{self.device.type}.amp.autocast")():
                    loss, correct, total = self.forward_pass(inputs, labels)
            else:
                loss, correct, total = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])
            losses.append(loss.item())
            # if self.use_wandb:
            #     wandb.log({"lr": lrs[-1], "loss": losses[-1]})
        increase_indices = [
            i for i in range(1, len(losses)) if losses[i] - losses[i - 1] > 0
        ]
        if increase_indices:
            min_loss_idx = increase_indices[0] - 1
        else:
            min_loss_idx = np.argmin(losses)
        best_lr = lrs[min_loss_idx]
        self.model.load_state_dict(orig_model_state_dict)
        self.optimizer.load_state_dict(orig_opt_state_dict)
        table = list(zip(lrs, losses))
        print(tabulate(table, headers=["Learning Rate", "Loss"], tablefmt="pretty"))
        return best_lr
