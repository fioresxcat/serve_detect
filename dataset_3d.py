import numpy as np
import pdb
from easydict import EasyDict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import torch
import pytorch_lightning as pl
from PIL import Image
import time
from turbojpeg import TurboJPEG
import albumentations as A
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


def load_from_pickle(fp):
    with open(fp, 'rb') as f:
        bin = f.read()
        obj = pickle.loads(bin)
    return obj



class ServeDataset(Dataset):
    def __init__(
        self,
        data_path,
        transforms, 
        mode,
        masked,
        n_input_frames,
        n_sample_limit,
        crop_size,
        do_augment=False,
        augment_props={},
    ):
        super(ServeDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.do_augment = do_augment
        self.augment_props = augment_props
        self.orig_h, self.orig_w = 1080, 1920
        self.n_input_frames = n_input_frames
        self.n_sample_limit = int(n_sample_limit)
        self.crop_size = crop_size
        self.transforms = transforms
        self.masked = masked
        self.jpeg_reader = TurboJPEG()
        self.normalize_video = NormalizeVideo(
            mean = [0.45, 0.45, 0.45],
            std = [0.225, 0.225, 0.225]
        )
        self.label2id = {
            'serve': 1,
            'no_serve': 0
        }
        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        self.data_items = load_from_pickle(self.data_path)
        


    def __len__(self):
        return len(self.data_items)
    

    def __getitem__(self, index):
        item = self.data_items[index]
        img_paths, label = item['img_paths'], item['label']
        img_paths = img_paths[:self.n_input_frames]  # 15 by default

        input_imgs = []
        for fp in img_paths:
            with open(fp, 'rb') as in_file:
                resized_img = cv2.resize(self.jpeg_reader.decode(in_file.read(), 0), (self.crop_size[0], self.crop_size[1]))  # already rgb images
            input_imgs.append(resized_img)
        if len(input_imgs) < self.n_input_frames:
            input_imgs.extend([input_imgs[-1]] * (self.n_input_frames - len(input_imgs)))

        if self.masked:
            txt_paths = [str(Path(img_fp).with_suffix('.txt')) for img_fp in img_paths]
            for i, txt_fp in enumerate(txt_paths):
                with open(txt_fp, 'r') as f:
                    lines = f.readlines()
                img = input_imgs[i]
                mask = np.zeros_like(img)[:, :, 0]
                for line in lines:
                    cl, x, y, w, h = line.strip().split()[:5]
                    cl = int(cl)
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if cl == 1:
                        xmin = (x - w/2) * img.shape[1]
                        ymin = (y - h/2) * img.shape[0]
                        xmax = (x + w/2) * img.shape[1]
                        ymax = (y + h/2) * img.shape[0]
                        mask[ymin:ymax, xmin:xmax, :] = 1
                    elif cl == 0:
                        pts = [float(el) for el in line.strip().split()[5:]]
                        pts = np.array(pts).reshape(-1, 2) * np.array([img.shape[1], img.shape[0]])
                        pts = pts.astype(np.int32)
                        mask = cv2.drawContours(mask, [pts], -1, 255, -1)
                input_imgs[i] = cv2.bitwise_and(img, img, mask=mask)



        if self.mode == 'train' and np.random.rand() < self.augment_props.augment_img_prob:
            transformed = self.transforms(
                image=input_imgs[0],
                image0=input_imgs[1],
                image1=input_imgs[2],
                image2=input_imgs[3],
                image3=input_imgs[4],
                image4=input_imgs[5],
                image5=input_imgs[6],
                image6=input_imgs[7],
                image7=input_imgs[8],
                image8=input_imgs[9],
                image9=input_imgs[10],
                image10=input_imgs[11],
                image11=input_imgs[12],
                image12=input_imgs[13],
                image13=input_imgs[14],
            )
            transformed_imgs = [transformed[k] for k in sorted([k for k in transformed.keys() if k.startswith('image')])]
            transformed_imgs = np.stack(transformed_imgs, axis=0)
        else:
            transformed_imgs = np.stack(input_imgs, axis=0)   # shape 15 x h x w x 3

        # normalize
        transformed_imgs = torch.from_numpy(transformed_imgs)
        transformed_imgs = transformed_imgs.permute(3, 0, 1, 2) # shape 3 x 15 x h x w
        transformed_imgs = transformed_imgs / 255.0
        transformed_imgs = self.normalize_video(transformed_imgs)

        if self.mode == 'predict':
            return img_paths, transformed_imgs, torch.tensor(self.label2id[label], dtype=torch.float)
        else:
            return transformed_imgs, torch.tensor([self.label2id[label]], dtype=torch.float)



class ServeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path, 
        test_path,
        predict_path,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(ServeDataModule, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.data_cfg = EasyDict(data_cfg)
        self.training_cfg = EasyDict(training_cfg)

        if self.data_cfg.do_augment:
            add_target = {
                'image0': 'image', 
                'image1': 'image', 
                'image2': 'image', 
                'image3': 'image', 
                'image4': 'image', 
                'image5': 'image', 
                'image6': 'image', 
                'image7': 'image',
                'image8': 'image',
                'image9': 'image',
                'image10': 'image',
                'image11': 'image',
                'image12': 'image',
                'image13': 'image',
            }
            
            self.transforms = A.Compose(
                A.SomeOf([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.ColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, always_apply=False),
                    A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=0),
                ], n=2),
                additional_targets=add_target,
            )
        else:
            self.transforms = None

    
    def setup(self, stage):
        if stage == 'fit' or stage == 'validate':
            self.train_ds = ServeDataset(data_path=self.train_path, transforms=self.transforms, mode='train', **self.data_cfg)
            self.val_ds = ServeDataset(data_path=self.val_path, transforms=None, mode='val', **self.data_cfg)
        elif stage == 'test':
            self.test_ds = ServeDataset(self.test_path, transforms=None, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = ServeDataset(self.predict_path, transforms=None, mode='predict', **self.data_cfg)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=self.training_cfg.shuffle_train, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    with open('config_3d.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    ds = ServeDataset(
        data_path=config.data.test_path,
        transforms=None, 
        mode='test',
        **config.data.data_cfg
    )
    ds_loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    for i, item in enumerate(ds_loader):
        imgs, label = item
        print(imgs.shape)
        print(label.shape)
        break
    pdb.set_trace()
    print('ok')