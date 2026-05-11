import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging


# ==============================
# 1. 参数解析
# ==============================
def get_args():
    parser = argparse.ArgumentParser(description="Generic Deep Learning Training Script")

    # 必需参数（根据您的命令行）
    parser.add_argument('--device', type=str, required=True, help='Device to use: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')

    # 可选参数（您可以根据需要添加或修改）
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs (checkpoints, logs)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')

    return parser.parse_args()


# ==============================
# 2. 模型定义（示例：使用ResNet-50 + DeepLabV3）
# ==============================
def build_model(pretrained=True):
    # 这里以 torchvision 的 DeepLabV3 为例
    # 请根据您的实际模型替换
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(pretrained=pretrained, num_classes=4)  # EchoNet-Dynamic 通常是4类
    return model


# ==============================
# 3. 数据集和数据加载器
# ==============================
def get_dataloaders(data_dir, batch_size, num_workers):
    # 示例：简单的图像变换（请根据您的数据集调整）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: 替换为您的实际 Dataset 类
    from torchvision.datasets import VOCSegmentation  # 仅作示例

    train_dataset = VOCSegmentation(root=data_dir, year='2012', image_set='train', download=False, transform=transform,
                                    target_transform=transform)
    val_dataset = VOCSegmentation(root=data_dir, year='2012', image_set='val', download=False, transform=transform,
                                  target_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ==============================
# 4. 训练函数
# ==============================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']  # DeepLabV3 输出是 dict
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(dataloader)
    logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")
    return avg_loss


# ==============================
# 5. 验证函数
# ==============================
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    logging.info(f"Epoch {epoch + 1} - Validation Loss: {avg_loss:.4f}")
    return avg_loss


# ==============================
# 6. 主函数
# ==============================
def main():
    args = get_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Training started with args: {args}")

    # 构建模型
    model = build_model(pretrained=args.pretrained)
    model.to(device)

    # 数据加载器
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"Best model saved at epoch {epoch + 1}")

        # 保存最后模型
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model.pth'))

    print("Training completed.")
    logging.info("Training completed.")


if __name__ == '__main__':
    main()