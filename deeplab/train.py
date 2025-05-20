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

downsample_size = (256,256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["rural", "urban"], required=True, help="Dataset to use")
    parser.add_argument("--batchsize", type=int, required=False, default=10)
    parser.add_argument("--numepoch", type=int, required=False, default=20)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--lrdecay", type=str,choices=["none", "step", "poly"], required=False, default="none")
    args = parser.parse_args()

    base_path =os.path.join("..", "Dataset")

    

    train_dataset_path = os.path.join(base_path,"Train",args.dataset.capitalize())
    #val_dataset_path = os.path.join(base_path, "Val", "Dataset", args.dataset.capitalize())
    print(f"loading dataset from {train_dataset_path} ...")
    transform = transforms.Compose([
    transforms.Resize(downsample_size),
    transforms.ToTensor()
    ])

    train_dataset = loveDAcustom(os.path.join(train_dataset_path,"images_png"), os.path.join(train_dataset_path,"masks_png"),transform)

    print(f"load {len(train_dataset)} images")

    
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,num_workers=1)

    model = DeepLabV2(n_classes=7+1).to(device)  # 0 (ignore), 1-6 (class)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    lr = args.lr

    enablePrint=0
    

    for epoch in range(args.numepoch):
        tq = tqdm(total=len(train_loader) * args.batchsize)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        model.train()
        total_loss = 0

        for images, masks in train_loader:

            images = images.to(device)              
            masks = masks.to(device)  

            optimizer.zero_grad()

            outputs = model(images)

            #YET TO FIX
            # Downsample prediction & mask to 1/8 of the real image → 128×128
            #outputs_ds = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=False)
            masks_ds = F.interpolate(masks.unsqueeze(1).float(), size=downsample_size, mode='nearest').squeeze(1).long()

            if enablePrint:
                print(masks_ds.size())
            if enablePrint:
                print("computing loss")
            loss = criterion(outputs, masks_ds)

            loss.backward()
            if enablePrint:
                print("step")
            optimizer.step()

            total_loss += loss.item()
            tq.update(args.batchsize)
            tq.set_postfix(loss='%.6f' % loss)
        tq.close()

        print(f"Epoch {epoch+1}/{args.numepoch}, Loss: {total_loss:.4f}")