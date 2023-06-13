import numpy as np
import pdb
from easydict import EasyDict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
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
        n_input_frames,
        n_sample_limit,
        crop_size,
        already_cropped,
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
        self.already_cropped = already_cropped
        self.transforms = transforms
        self.jpeg_reader = TurboJPEG()

        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        data_dict = load_from_pickle(self.data_path)
        self.ls_img_paths = sorted(data_dict.keys())[:int(self.n_sample_limit)]
        self.ls_labels = [data_dict[img_paths] for img_paths in self.ls_img_paths]


    def __len__(self):
        return len(self.ls_img_paths)
    

    def __getitem__(self, index):
        item = self.data_items[index]
        img_paths, yolo_anno_paths, label = item['img_paths'], item['yolo_anno_paths'], item['label']
        
        # process img
        if self.already_cropped:
            input_imgs, ls_norm_pos = self.get_already_cropped_images(img_paths, ls_norm_pos)
        else:
            input_imgs, ls_norm_pos = self.crop_images_from_paths(img_paths, ls_norm_pos)
        

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
            )
            transformed_imgs = [transformed[k] for k in sorted([k for k in transformed.keys() if k.startswith('image')])]
            transformed_imgs = np.concatenate(transformed_imgs, axis=2)
            transformed_imgs = torch.tensor(transformed_imgs)
        else:
            transformed_imgs = torch.tensor(np.concatenate(input_imgs, axis=2))

        # normalize
        transformed_imgs = transformed_imgs.permute(2, 0, 1) / 255.

        # construct event target
        if event_target[0] != 0:
            event_target = torch.tensor([event_target[0], 0, 1-event_target[0]], dtype=torch.float)
        elif event_target[1] != 0:
            event_target = torch.tensor([0, event_target[1], 1-event_target[1]], dtype=torch.float)
        else:
            event_target = torch.tensor([0, 0, 1], dtype=torch.float)

        if self.mode != 'predict':
            return transformed_imgs, torch.tensor(ls_norm_pos), event_target
        else:
            return img_paths, transformed_imgs, torch.tensor(ls_norm_pos), event_target




class EventDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path, 
        test_path,
        predict_path,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(EventDataModule, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.data_cfg = EasyDict(data_cfg)
        self.training_cfg = EasyDict(training_cfg)

        if self.data_cfg.do_augment:
            if self.data_cfg.n_input_frames == 3:
                add_target = {'image0': 'image', 'image1': 'image'}
            elif self.data_cfg.n_input_frames == 5:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}
            elif self.data_cfg.n_input_frames == 9:
                add_target = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image', 'image5': 'image', 'image6': 'image', 'image7': 'image'}
            
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

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    ds = ServeDataset(
        data_path=config.data.test_path,
        transforms=None, 
        mode='test',
        **config.data.data_cfg
    )

    for i, item in enumerate(ds):
        imgs, pos, ev = item
        print(imgs.shape)
        print(pos.shape)
        print(ev.shape)
        break
    pdb.set_trace()
    print('ok')