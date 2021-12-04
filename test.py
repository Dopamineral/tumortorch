import argparse
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch import unsqueeze


working_directory = 'C:/Users/rober/Desktop'
os.chdir(working_directory)
checkpoint_folder = 'checkpoints_04-12-2021_12.39.15'
checkpoint_path = os.path.join(working_directory,checkpoint_folder)
os.chdir(checkpoint_path)
checkpoint_file = 'checkpoint_0860.pt'
checkpoint = torch.load(checkpoint_file)

# load model
model_checkpoint = models.resnet18(pretrained=True)
model_checkpoint.fc = nn.Linear(512,2) 

model_checkpoint.load_state_dict(checkpoint['model_state_dict'])


# load loss, accuracy and plot them. AWESOME it works.
loss = checkpoint['loss']
val_accuracy = checkpoint['val_accuracy']
plt.plot(loss)
plt.plot(val_accuracy)

trans1 = transforms.Resize([224,224])
trans2 = transforms.Grayscale(num_output_channels=3)
trans3 = transforms.RandomHorizontalFlip(p=0.5)
trans4 = transforms.RandomVerticalFlip(p=0.5)
trans5 = transforms.PILToTensor()
trans6 = transforms.ConvertImageDtype(torch.float)

mean = 0 #for normalization transform 
std = 1
trans7 = transforms.Normalize(mean,std)

image_transforms = transforms.Compose([trans1,trans2,trans3,trans4,trans5,trans6,trans7])


class imagetestset(Dataset):

    def __init__(self,csv_file,csv_dir,transform=None): 

        self.transform = transform
        self.data_dir = csv_dir
        datadir_csv_file = os.path.join(self.data_dir, csv_file)
        self.data_name = pd.read_csv(datadir_csv_file,index_col=False)
    
        self.X = self.data_name.iloc[:,0] #image path
        self.y = self.data_name.iloc[:,1] #image label
        LE = preprocessing.LabelEncoder()
        self.y = LE.fit_transform(self.y)
        self.y = torch.as_tensor(self.y,dtype=torch.long)
        
        self.len = self.X.shape[0]
                    
        
    def __len__(self):
        return self.len
        
    def __getitem__(self,index):
        xout = self.X.iloc[index]
        image = Image.open(xout) #loads image with PIL
        
        yout = self.y[index]
        
        if self.transform:
            image = self.transform(image) #apply transformations
            
        return image, yout
    


dataset = imagetestset('data.csv', 'C:/Users/rober/Desktop',transform=image_transforms)
def pred_dataset(dataset,sample):   
    sm = nn.Softmax()
    transformed_image = dataset[sample][0]
    transformed_image = transformed_image.unsqueeze(0)
    
    model_checkpoint.eval()
    prediction = model_checkpoint(transformed_image)
    probability = sm(prediction).detach().numpy()
    prob_0 = probability[0][0]
    prob_1 = probability[0][1]
    
    if prob_0 > prob_1:
        prediction = 0
    else:
        prediction = 1

    
    
    return prob_0, prob_1, prediction


prob = pred_dataset(dataset,1)

image_folder = os.path.join(working_directory,'images_predicted')
os.chdir(image_folder)

for ii in range(dataset.len):
    print(ii)
    prob_0, prob_1, pred = pred_dataset(dataset,ii)
    
    actual_index = dataset[ii][1]
    if actual_index == 1:
        actual = "tumor"
    else:
        actual="normal"
    
    fig = plt.imshow(dataset[ii][0].numpy()[0], cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    
    if pred == 0:
        plt.title(f'Predicted: Normal. probability:{prob_0:.4f} \n actual: {actual}')
    else:
        plt.title(f'Predicted: Tumor. probability:{prob_1:.4f}  \n actual: {actual}')
    plt.show()
    plt.savefig(f'image_{ii}.png')
    
    
    


