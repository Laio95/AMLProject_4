import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class LoveDA(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=8,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(LoveDA, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0: ignore_label, 1: 1,
                              2: 2, 3: 3,
                              4: 4, 5: 5,
                              6: 6 , 7:7}
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            if 'test' in self.list_path:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "name": name
                })
            else:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
            
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root,'camvid',item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        color_map = Image.open(os.path.join(self.root,'camvid',item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))