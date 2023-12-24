import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.data_folder = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_path,self.data_folder)
        #self.img_file = self._select_img(self.data_file)
    #questo devo fare in modo che avvenga su una serie di sotto caretlle, perchè ha 
    #difficoltà a lavorare su una caretlla unica    
    
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
                
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    #def __getitem__(self, idx):
    #    img_file = self.img_file[idx]
    #    with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
    #        img = torch.from_numpy(pickle.load(file)).float()
    #        # print(img.shape)
    #    gt_file = "gt" + img_file[3:]
    #    with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
    #        gt = torch.from_numpy(pickle.load(file)).float()
    #        # gt = torch.unsqueeze(gt, dim=0) ####
    #        # print(gt.shape)

    #funzione place holder data dal fatto che drive è coglione
    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        try:
            with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
                img = torch.from_numpy(pickle.load(file)).float()
        except FileNotFoundError:
            print(f"Image file '{img_file}' not found.")
            # You can handle this missing file scenario by skipping or returning a placeholder tensor
            # For example:
            img = torch.zeros((1,256,256), dtype=torch.float)  # Placeholder tensor

        folder=os.path.dirname(img_file)
        name=os.path.basename(img_file)
        gt_file = os.path.join(folder,"gt" + name[3:])
        try:
            with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
                gt = torch.from_numpy(pickle.load(file)).float()
        except FileNotFoundError:
            print(f"Ground truth file '{gt_file}' not found.")
            gt = torch.zeros((1,256,256), dtype=torch.float)
            ###
        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt

    #def _select_img(self, file_list):
    #  img_list = []
    #  for file in file_list:
    #   if file[:3] == "img" :
    #      img_list.append(file)
    def _select_img(self,data_path, folder_list):
       
      img_list = []
      for folder in folder_list:
          file_list=os.listdir(os.path.join(data_path,folder))
      
      
          for file in file_list:
              if file[:3] == "img" :
                  img_list.append(os.path.join(folder,file))
#
      return img_list

    def __len__(self):
        return len(self.img_file)
