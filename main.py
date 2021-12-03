import argparse
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms



#input arguments run this as python main.py --seed 121 for example. 
parser = argparse.ArgumentParser(description='DeepSeeker model Training')
parser.add_argument('--seed', default=0, type=int,
                    help='set custom seed for initializing training')

parser.add_argument('--testratio',default = 0.3, type=float,
                    help='Define custom test size ratio')
opts = parser.parse_args()

torch.manual_seed(opts.seed)

test_ratio = opts.testratio
#%% dataset
class imagedataset(Dataset):

    def __init__(self,csv_file,csv_dir,train=True,transform=None): 

        self.transform = transform
        self.data_dir = csv_dir
        datadir_csv_file = os.path.join(self.data_dir, csv_file)
        self.data_name = pd.read_csv(datadir_csv_file)
    
        self.X = self.data_name.iloc[:,0] #image path
        self.y = self.data_name.iloc[:,1] #image label
        self.len = self.X.shape[0]
        
        self.X_train, self.X_test, self.y_train, self.y_test =train_test_split(self.X, self.y, test_size=0.33, random_state=123) 
       
        
        if train==True:
            self.xout = self.X_train
            self.yout = self.y_train
            self.len = self.xout.shape[0]
            
        else:
            self.xout = self.X_test
            self.yout = self.y_test
            self.len = self.xout.shape[0]
            
        
    def __len__(self):
        return self.len
        
    def __getitem__(self,index):
        xout = self.xout.iloc[index]
        image = Image.open(xout)
        
        yout = self.yout[index]
        
        if self.transform:
            image = self.transform(image)
            
        return image, yout

#image transforms 
trans1 = transforms.Resize([200,200])
trans2 = transforms.RandomHorizontalFlip(p=0.5)
trans3 = transforms.RandomVerticalFlip(p=0.5)
trans4 = transforms.PILToTensor()
trans5 = transforms.ConvertImageDtype(torch.float)
trans6 = transforms.Normalize(0,1)

image_transforms = transforms.Compose([trans1,trans2,trans3,trans4,trans5,trans6])

dataset = imagedataset('data.csv', 'C:/Users/rober/Desktop',train=True,transform=image_transforms)
#%% dataloader
trainloader = DataLoader(imagedataset('data.csv', 'C:/Users/rober/Desktop',train=True,transform=image_transforms))
testloader = DataLoader(imagedataset('data.csv', 'C:/Users/rober/Desktop',train=False,transform=image_transforms))

# model

# training loop

# checkpoint handling.