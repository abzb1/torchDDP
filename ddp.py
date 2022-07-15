import torch as torch
from torch import distributed as dist

import sys
import os
import time

#import modules
import checkinit as chint
import vgg16bn
import dataloader as loader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

def run_ddp():
    rank = int(os.environ["SLURM_PROCID"])
    size = 2
    chint.setup("nccl", rank, size)
    
    batch_size = 128
    train_dataloader, _ = loader.cifar10data(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    epochs = 20

    model = vgg16bn.vgg16bn().to("cuda:0")
    ddp_model = DDP(model)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr = learning_rate)

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    writer = SummaryWriter("./runs/ddp_view_cifar10_vgg16D")
    running_loss = 0.0
    for t in range(epochs):
        print(f"epoch : {t}\n")
        for batch, (X, y) in enumerate(train_dataloader):
            model_in = X.to("cuda:0")
            label = y.to("cuda:0")

            optimizer.zero_grad()

            pred = ddp_model(model_in)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch % 10 == 9:
                writer.add_scalar("training loss", running_loss / 10, t * len(train_dataloader) + batch)
                running_loss = 0.0
                loss, current = loss.item(), batch * batch_size
                print(f"{rank}, loss : {loss:>7f} [{current:>5d}/{size:>5d}]")

    torch.save(model.state_dict(), "ddp_model_weights.pth")

    chint.cleanup()

if __name__ == "__main__":
    run_ddp()
