import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from lovedacustom import loveDAcustom
from deeplabv2 import DeepLabV2
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualizeDataset import show_prediction_and_groundtruth

from crf import DenseCRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7+1

# downsample size and batchsize can greatly vary the memory usage of the model.
# (256,256) with batchsize 3 does fit in 8GB of VRAM
downsample_size = (256,256)


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	lr = init_lr * (1 - iter / max_iter) ** power
	optimizer.param_groups[0]['lr'] = lr
	
	return lr

crf = DenseCRF(
    iter_max=10,
    pos_w=3,
    pos_xy_std=1,
    bi_w=4,
    bi_xy_std=67,
    bi_rgb_std=3
    )

def postprocess_crf(image,prediction):
    return crf(image,prediction)
    
def validate(model, dataset):
     model.eval()

     total_images=0

     with torch.no_grad():
        for images, masks in dataset:
            images = images.to(device)
            masks = masks.to(device)
            if(len(images.shape)>3):
                batch_size = images.size(0)
            else:
                batch_size = 1
            print(f"batch_size = {batch_size}")
            total_images += batch_size
            for i in range(batch_size):
                if( batch_size>1 ):
                    single_image = images[i].to(device)
                    single_mask = masks[i]
                else:
                    single_image = images
                    single_mask = masks
                
                
                start_time = time.time()
                # Compute prediction for each single image
                output = model(single_image.unsqueeze(0))
                prob = F.softmax(output[0], dim=0).cpu().numpy()
                single_image=(single_image.cpu().numpy()*255).clip(0, 255).astype(np.uint8).transpose(1,2,0)
                print(f"image {single_image.shape}")
                print(type(single_image), single_image.dtype)
                print(f"prob {prob.shape}")
                print(type(prob),prob.dtype)
                post_prob = postprocess_crf(single_image,prob)

                pred = np.argmax(post_prob, axis=0)  # (H, W)
                
                print(f"time taken for one image: {time.time()-start_time}")

                show_prediction_and_groundtruth(single_image,single_mask.cpu().numpy(),pred)
                #model.train()
                #return #debug only, just do one image


     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["rural", "urban"], required=True, help="Dataset to use")
    parser.add_argument("--batchsize", type=int, required=False, default=3)
    parser.add_argument("--numepoch", type=int, required=False, default=20)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--lrdecay", type=str,choices=["none", "step", "poly"], required=False, default="none")
    parser.add_argument("--modelname",type=str,required=False, default="deeplab_model")
    parser.add_argument("--test",type=bool,required=False, default=False)
    args = parser.parse_args()

    base_dataset_path =os.path.join("..", "Dataset")

    model_path = os.path.join("trainedModels",args.modelname+".pth")
    if not os.path.exists("trainedModels"):
        os.makedirs("trainedModels")

    

    train_dataset_path = os.path.join(base_dataset_path,"Train",args.dataset.capitalize())
    val_dataset_path = os.path.join(base_dataset_path, "Val", args.dataset.capitalize())
    print(f"loading dataset from {train_dataset_path} ...")
    transform = transforms.Compose([
    transforms.Resize(downsample_size),
    transforms.ToTensor()
    ])

    train_dataset = loveDAcustom(os.path.join(train_dataset_path,"images_png"), os.path.join(train_dataset_path,"masks_png"),transform)
    val_dataset = loveDAcustom(os.path.join(val_dataset_path,"images_png"), os.path.join(val_dataset_path,"masks_png"),transform)
    print(f"loaded {len(train_dataset)} images")

    
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=1)

    model = DeepLabV2(n_classes=NUM_CLASSES).to(device)  # 0 (ignore), 1-7 (class)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    if not os.path.isfile(model_path):
        for epoch in range(args.numepoch):
            if args.lrdecay == "poly":
                lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.numepochs)
            else:
                lr = args.lr
            tq = tqdm(total=len(train_loader) * args.batchsize)
            tq.set_description('epoch %d, lr %f' % (epoch, lr))
            model.train()
            total_loss = 0
            total_loss_n = 0

            for images, masks in train_loader:

                images = images.to(device)              
                masks = masks.to(device)  

                optimizer.zero_grad()

                outputs = model(images)

                #YET TO FIX
                # Downsample prediction & mask to 1/8 of the real image → 128×128
                #outputs_ds = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=False)
                masks_ds = F.interpolate(masks.unsqueeze(1).float(), size=downsample_size, mode='nearest').squeeze(1).long()


                loss = criterion(outputs, masks_ds)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()
                total_loss_n +=1
                tq.update(args.batchsize)
                tq.set_postfix(loss='%.6f' % loss)
            tq.close()
            print(f"Epoch {epoch+1}/{args.numepoch}, AVG Loss: {(total_loss/total_loss_n):.4f}")
            #validate(model,train_dataset)
        torch.save(model.state_dict(),model_path)
        print(f"model saved as {model_path}")
    else:
        print("model exist, validating:")
        model.load_state_dict(torch.load(model_path, map_location=device))
        validate(model,val_dataset)
    

        