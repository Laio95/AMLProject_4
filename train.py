import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from lovedacustom import loveDAcustom
from deeplabv2 import DeepLabV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["rural", "urban"], required=True, help="Dataset to use")
    parser.add_argument("--batchsize", type=int, required=False, default=1)
    parser.add_argument("--numepoch", type=int, required=False, default=20)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--lrdecay", type=str,choices=["none", "step", "poly"], required=False, default="none")
    args = parser.parse_args()

    base_path =os.path.join("..", "Dataset")

    

    train_dataset_path = os.path.join(base_path,"Train",args.dataset.capitalize())
    #val_dataset_path = os.path.join(base_path, "Val", "Dataset", args.dataset.capitalize())
    print(f"loading dataset from {train_dataset_path} ...")
    transform = transforms.ToTensor()

    train_dataset = loveDAcustom(os.path.join(train_dataset_path,"images_png"), os.path.join(train_dataset_path,"masks_png"),transform)

    print(f"load {len(train_dataset)} images")

    
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    model = DeepLabV2(n_classes=7).to(device)  # 0 (ignore), 1-6 (class)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    lr = args.lr
    

    for epoch in range(args.numepoch):
        tq = tqdm(total=len(train_loader) * args.batchsize)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        model.train()
        total_loss = 0

        for images, masks in train_loader:
            print("image to device")
            images = images.to(device)               # (B, 3, 1024, 1024)
            masks = masks.to(device)                 # (B, 1024, 1024)
            print("zero grad")
            optimizer.zero_grad()
            print("computing output")
            outputs = model(images)                  # (B, C, 1024, 1024)
            print("interpolating output")
            # Downsample prediction & mask to 1/8 of the real image → 128×128
            outputs_ds = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=False)
            masks_ds = F.interpolate(masks.unsqueeze(1).float(), size=(128, 128), mode='nearest').squeeze(1).long()
            print("computing loss")
            loss = criterion(outputs_ds, masks_ds)
            print("backward")
            loss.backward()
            print("step")
            optimizer.step()

            total_loss += loss.item()
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
        tq.close()

        print(f"Epoch {epoch+1}/{args.numepoch}, Loss: {total_loss:.4f}")