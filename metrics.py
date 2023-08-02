# ## Define Metrics - Utility class to measure accuracy of the model and plot metrics
from config import torch


class Metrics:
    """class that holds logic for calculating accuracy and printing it"""

    def __init__(self):
        self.acc = {"train": [], "val": []}
        self.loss = {"train": [], "val": []}

    @staticmethod
    @torch.no_grad()
    def accuracy(yhat, labels, debug):
        """accuracy of a batch"""
        yhat = torch.log_softmax(yhat, dim=1)  # softmax of logit output
        yhat = yhat.max(1)[1]  # get index of max values
        if debug:
            print(f"outputs: {yhat} labels: {labels}")
            print(f" output == label ?: {torch.equal(yhat, labels)}")
        acc = yhat.eq(labels).sum() / len(yhat)
        return acc

    def __str__(self):
        return (
            f"loss:\n training set  : {self.loss['train'][-1]:.4}\n validation set: {self.loss['val'][-1]:.4}\n"
            f"accuracy:\n training set  : {self.acc['train'][-1]:.4}\n validation set: {self.acc['val'][-1]:.4} "
        )

    def plot(self):
        """plot loss and acc curves"""
        if plt.get_backend() == "agg":
            print(
                "Average training accuracy score: {sum(acc['train'])/len(acc['train'])}"
            )
            print(
                "Average validation accuracy score: {sum(acc['val'])/len(acc['val'])}"
            )
        else:
            train_acc = [x * 100 for x in self.acc["train"]]
            val_acc = [x * 100 for x in self.acc["val"]]
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 2.5))
            ax[0].plot(self.loss["train"], "-o")
            ax[0].plot(self.loss["val"], "-o")
            ax[0].set_ylabel("loss")
            ax[0].set_title(f"Train vs validation loss")
            ax[1].plot(train_acc, "-o")
            ax[1].plot(val_acc, "-o")
            ax[1].set_ylabel("accuracy (%)")
            ax[1].set_title("Training vs validation acc")
            for x in ax:
                x.yaxis.grid(True)
                x.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                x.legend(["train", "validation"])
                x.set_xlabel("epoch")
            plt.show()
