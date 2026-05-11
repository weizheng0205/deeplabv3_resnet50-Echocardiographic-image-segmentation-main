import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Subset, DataLoader

from utils.echo import Echo
from utils.utils import get_mean_and_std, bootstrap, dice_similarity_coefficient, latexify, savevideo


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default='data/EchoNet-Dynamic')
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if
           name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
              default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--save_video/--skip_video", default=True)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=0)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(data_dir=None, output=None, model_name="deeplabv3_resnet50", pretrained=False, weights=None, run_test=False,
        save_video=False, num_epochs=50, lr=1e-5, weight_decay=1e-5, lr_step_period=None, num_train_patients=None,
        num_workers=0, batch_size=20, device=None, seed=0):
    """

    :param data_dir: 包含数据集的目录
    :param output: 存放输出的目录名
    :param model_name: 分割模型的名称。"deeplabv3_resnet50"，"deeplabv3_resnet101"
    :param pretrained: 是否为模型使用预训练的权重
    :param weights: 包含初始化模型权重的检查点的路径
    :param run_test: 是否运行测试
    :param save_video: 是否保存带有分段的视频
    :param num_epochs: 训练期间的epoch数量
    :param lr: SGD的学习率
    :param weight_decay: SGD的权重衰减
    :param lr_step_period: 学习率衰减的周期 math.Inf（永不衰减学习率）
    :param num_train_patients: 消融
    :param num_workers: windows不要动
    :param batch_size: 每批要加载多少个样本
    :param device: 运行的设备
    :param seed: 随机数生成器的种子

    :return:
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation",
                              "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 添加GPU信息输出
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Set up model - 修复新版torchvision API问题
    model = None
    if weights is not None:
        # 如果提供了权重文件，直接加载模型结构
        print(f"加载现有权重: {weights}")
        try:
            # 先创建基础模型
            model = torchvision.models.segmentation.__dict__[model_name](pretrained=False)
            model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1,
                                                   kernel_size=model.classifier[-1].kernel_size)

            # 加载权重
            checkpoint = torch.load(weights, map_location=device, weights_only=True)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("权重加载成功!")

        except Exception as e:
            print(f"加载权重失败: {e}")
            return
    else:
        # 创建新模型
        try:
            # 新版本API
            if pretrained:
                print("使用预训练权重...")
                weights_obj = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                model = torchvision.models.segmentation.__dict__[model_name](weights=weights_obj)
                #封装了ASPP模块
            else:
                print("使用随机初始化权重...")
                model = torchvision.models.segmentation.__dict__[model_name](weights=None)
        except (TypeError, AttributeError):
            # 旧版本API兼容
            try:
                model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)
            except:
                # 如果还不行，使用最基础的初始化
                model = torchvision.models.segmentation.__dict__[model_name](pretrained=False)

        # 修改输出层
        model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1,
                                               kernel_size=model.classifier[-1].kernel_size)

    if str(device) == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # 因为这个数据集的特殊性，没法应用常规的均值和方差，所以需要自己算
    mean, std = get_mean_and_std(Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks, "mean": mean, "std": std}

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = Subset(dataset["train"], indices)
    dataset["val"] = Echo(root=data_dir, split="val", **kwargs)

    # 如果只是测试，跳过训练
    if weights is not None and not run_test:
        print("权重已加载，跳过训练，直接进行测试...")
        # 直接跳到测试部分
        run_test = True
        num_epochs = 0

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")

        # 如果有权重文件且不是从检查点恢复，跳过训练恢复
        if weights is None:
            try:
                # Attempt to load checkpoint
                checkpoint = torch.load(os.path.join(output, "checkpoint.pt"), map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])
                optim.load_state_dict(checkpoint['opt_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
                epoch_resume = checkpoint["epoch"] + 1
                bestLoss = checkpoint["best_loss"]
                f.write("Resuming from epoch {}\n".format(epoch_resume))
            except FileNotFoundError:
                f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                        pin_memory=(str(device) == "cuda"), drop_last=(phase == "train"))

                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader,
                                                                                     phase == "train",
                                                                                     optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (
                        large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())

                # 获取GPU内存使用信息
                if torch.cuda.is_available():
                    max_allocated = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                    max_reserved = sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count()))
                else:
                    max_allocated, max_reserved = 0, 0

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch, phase, loss, overall_dice, large_dice,
                                                                    small_dice, time.time() - start_time,
                                                                    large_inter.size,
                                                                    max_allocated,
                                                                    max_reserved,
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': bestLoss,
                    'loss': loss,
                    'opt_dict': optim.state_dict(),
                    'scheduler_dict': scheduler.state_dict()}
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            print("开始测试...")
            # Run on validation and test
            for split in ["val", "test"]:
                print(f"测试 {split} 集...")
                dataset_split = Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset_split,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                                         pin_memory=(str(device) == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, False, None,
                                                                                     device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)

                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset_split.fnames, overall_dice, large_dice,
                                                                 small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                s1, s2, s3 = bootstrap(np.concatenate((large_inter, small_inter)),
                                       np.concatenate((large_union, small_union)), dice_similarity_coefficient)
                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                s1, s2, s3 = bootstrap(large_inter, large_union, dice_similarity_coefficient)
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                s1, s2, s3 = bootstrap(small_inter, small_union, dice_similarity_coefficient)
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                f.flush()

    # Saving videos with segmentations
    if save_video:
        print("生成分割视频...")
        dataset_video = Echo(root=data_dir, split="test", target_type=["Filename", "LargeIndex", "SmallIndex"],
                             mean=mean, std=std,
                             length=None, max_length=None, period=1
                             )
        dataloader = DataLoader(dataset_video, batch_size=10, num_workers=num_workers, shuffle=False,
                                pin_memory=False, collate_fn=_video_collate_fn)

        # Save videos with segmentation
        video_output_dir = os.path.join(output, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        if not all(os.path.isfile(os.path.join(video_output_dir, f)) for f in dataset_video.fnames):
            model.eval()

            os.makedirs(os.path.join(output, "size"), exist_ok=True)
            latexify()

            with torch.no_grad():
                with open(os.path.join(output, "size.csv"), "w") as g:
                    g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                    for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                        # 使用GPU进行预测
                        x = x.to(device)
                        y = torch.cat([model(x[i:(i + batch_size)])["out"] for i in range(0, x.shape[0], batch_size)],
                                      dim=0)
                        y = y.detach().cpu().numpy()

                        start = 0
                        x = x.cpu().numpy()
                        for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                            # 提取一个视频和分割预测
                            video = x[start:(start + offset), ...]
                            logit = y[start:(start + offset), 0, :, :]

                            # 反归一化视频
                            video *= std.reshape(1, 3, 1, 1)
                            video += mean.reshape(1, 3, 1, 1)

                            f, c, h, w = video.shape
                            assert c == 3

                            video = np.concatenate((video, video), 3)
                            video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])
                            video = np.concatenate((video, np.zeros_like(video)), 2)

                            size = (logit > 0).sum((1, 2))

                            trim_min = sorted(size)[round(len(size) ** 0.05)]
                            trim_max = sorted(size)[round(len(size) ** 0.95)]
                            trim_range = trim_max - trim_min
                            systole = set(
                                scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                            for (frame, s) in enumerate(size):
                                g.write(
                                    "{},{},{},{},{},{}\n".format(filename, frame, s,
                                                                 1 if frame == large_index[i] else 0,
                                                                 1 if frame == small_index[i] else 0,
                                                                 1 if frame in systole else 0))

                            fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                            plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                            ylim = plt.ylim()
                            for s in systole:
                                plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                            plt.ylim(ylim)
                            plt.title(os.path.splitext(filename)[0])
                            plt.xlabel("Seconds")
                            plt.ylabel("Size (pixels)")
                            plt.tight_layout()
                            plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                            plt.close(fig)

                            size -= size.min()
                            size = size / size.max()
                            size = 1 - size

                            for (f, s) in enumerate(size):
                                video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                                if f in systole:
                                    video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                                def dash(start, stop, on=10, off=10):
                                    buf = []
                                    x = start
                                    while x < stop:
                                        buf.extend(range(x, x + on))
                                        x += on
                                        x += off
                                    buf = np.array(buf)
                                    buf = buf[buf < stop]
                                    return buf

                                d = dash(115, 224)

                                if f == large_index[i]:
                                    video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array(
                                        [0, 225, 0]).reshape(
                                        (1, 3, 1))
                                if f == small_index[i]:
                                    video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array(
                                        [0, 0, 225]).reshape(
                                        (1, 3, 1))

                                r, c = skimage.draw.disk(
                                    (int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))),
                                    4.1)
                                video[f, :, r, c] = 255.

                            video = video.transpose(1, 0, 2, 3)
                            video = video.astype(np.uint8)
                            savevideo(os.path.join(video_output_dir, filename), video, 50)

                            start += offset

        print("所有处理完成!")


def run_epoch(model, dataloader, train, optim, device):
    """
    运行一个 epoch 的 训练 / 评估
    """
    total = 0.0
    n = 0
    model.train(train)

    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                large_frame = large_frame.to(device)
                small_frame = small_frame.to(device)
                large_trace = large_trace.to(device)
                small_trace = small_trace.to(device)

                # 对舒张期帧进行预测
                y_large = model(large_frame)["out"]
                loss_large = binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")

                large_pred = (y_large[:, 0, :, :] > 0).float()
                large_target = (large_trace > 0).float()

                large_inter = (large_pred * large_target).sum(dim=(1, 2))
                large_union = (large_pred + large_target).clamp(0, 1).sum(dim=(1, 2))

                large_inter_list.append(large_inter.detach().cpu())
                large_union_list.append(large_union.detach().cpu())

                # 对收缩期帧进行预测
                y_small = model(small_frame)["out"]
                loss_small = binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")

                small_pred = (y_small[:, 0, :, :] > 0).float()
                small_target = (small_trace > 0).float()

                small_inter = (small_pred * small_target).sum(dim=(1, 2))
                small_union = (small_pred + small_target).clamp(0, 1).sum(dim=(1, 2))

                small_inter_list.append(small_inter.detach().cpu())
                small_union_list.append(small_union.detach().cpu())

                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item()
                n += large_trace.size(0)

                current_loss = total / n / 112 / 112
                batch_loss = loss.item() / large_trace.size(0) / 112 / 112

                current_large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum() + 1e-8)
                current_small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum() + 1e-8)

                s = "平均损失:{:.4f} ({:.4f}), 舒张期Dice:{:.4f}, 收缩期Dice:{:.4f}".format(
                    current_loss, batch_loss, current_large_dice.item(), current_small_dice.item())
                pbar.set_description(s)
                pbar.update()

    large_inter_total = torch.cat(large_inter_list).numpy()
    large_union_total = torch.cat(large_union_list).numpy()
    small_inter_total = torch.cat(small_inter_list).numpy()
    small_union_total = torch.cat(small_union_list).numpy()

    loss = total / n / 112 / 112

    return loss, large_inter_total, large_union_total, small_inter_total, small_union_total


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos."""
    video, target = zip(*x)
    i = list(map(lambda t: t.shape[1], video))
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))
    target = zip(*target)
    return video, target, i


if __name__ == "__main__":
    run()