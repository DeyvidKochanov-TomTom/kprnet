import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from apex import parallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import semantic_kitti
from models import deeplab
import utils


parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--checkpoint-dir", required=True, type=Path)
args = parser.parse_args()


def run_val(model, val_loader, n_iter, writer):
    print("Runnign validation")
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    eval_metric = utils.evaluation.Eval(19, 255)
    with torch.no_grad():
        average_loss = 0
        for step, items in tqdm(enumerate(val_loader)):
            images = items["image"].cuda(0, non_blocking=True)
            labels = items["labels"].long().cuda(0, non_blocking=True)
            py = items["py"].float().cuda(0, non_blocking=True)
            px = items["px"].float().cuda(0, non_blocking=True)
            pxyz = items["points_xyz"].float().cuda(0, non_blocking=True)
            knns = items["knns"].long().cuda(0, non_blocking=True)
            predictions = model(images, px, py, pxyz, knns)

            loss = loss_fn(predictions, labels)
            average_loss += loss.item()
            _, predictions_argmax = torch.max(predictions, 1)
            eval_metric.update(predictions_argmax.cpu().numpy(), labels.cpu().numpy())

        average_loss /= step
        miou, ious = eval_metric.getIoU()
        print(f"Iteration {n_iter} Average Val Loss: {average_loss}, mIou {miou}")
        print(f"Per class Ious {ious}")
        writer.add_scalar("val/val", average_loss, n_iter)
        writer.add_scalar("val/mIoU", miou, n_iter)


def train(rank):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    dist.init_process_group(
        backend="nccl", init_method="tcp://localhost:34567", world_size=8, rank=rank
    )
    dist.barrier()

    model = deeplab.resnext101_aspp_kp(19)
    torch.cuda.set_device(rank)
    if rank == 0:
        writer = SummaryWriter(log_dir=args.checkpoint_dir, flush_secs=20)
    model = parallel.convert_syncbn_model(model)
    model.cuda(rank)

    model.load_state_dict(
        torch.load(
            args.model_dir / "resnext_cityscapes_2p.pth", map_location=f"cuda:{rank}"
        ),
        strict=False,
    )
    dist.barrier()
    if rank == 0:
        print(model.parameters)
    model = parallel.DistributedDataParallel(model)
    train_dataset = semantic_kitti.SemanticKitti(
        args.semantic_kitti_dir / "dataset/sequences", "train",
    )

    train_sampler = utils.dist_utils.TrainingSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=3,
        num_workers=8,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=semantic_kitti.SemanticKitti(
            args.semantic_kitti_dir / "dataset/sequences", "val",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    loss_fn = utils.ohem.OhemCrossEntropy(ignore_index=255, thresh=0.9, min_kept=10000)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-4
    )
    scheduler = utils.cosine_schedule.CosineAnnealingWarmUpRestarts(
        optimizer, T_0=96000, T_mult=10, eta_max=0.01875, T_up=1000, gamma=0.5
    )
    n_iter = 0
    for epoch in range(120):
        model.train()
        for step, items in enumerate(train_loader):
            images = items["image"].cuda(rank, non_blocking=True)
            labels = items["labels"].long().cuda(rank, non_blocking=True)
            py = items["py"].float().cuda(rank, non_blocking=True)
            px = items["px"].float().cuda(rank, non_blocking=True)
            pxyz = items["points_xyz"].float().cuda(rank, non_blocking=True)
            knns = items["knns"].long().cuda(rank, non_blocking=True)
            predictions = model(images, px, py, pxyz, knns)

            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            if rank == 0:
                print(
                    f"Epoch: {epoch} Iteration: {step} / {len(train_loader)} Loss: {loss.item()}"
                )
                writer.add_scalar("loss/train", loss.item(), n_iter)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], n_iter)
            n_iter += 1
            scheduler.step()

        if rank == 0:
            if (epoch + 1) % 5 == 0:
                run_val(model, val_loader, n_iter, writer)
            torch.save(
                model.module.state_dict(), args.checkpoint_dir / f"epoch{epoch}.pth"
            )


def main() -> None:
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, nprocs=ngpus)


if __name__ == "__main__":
    main()
