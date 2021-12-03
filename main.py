import argparse
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


#input arguments run this as python main.py --seed 121 for example. 
parser = argparse.ArgumentParser(description='DeepSeeker model Training')
parser.add_argument('--seed', default=0, type=int,
                    help='set custom seed for initializing training')

parser.add_argument('--testratio',default = 0.3, type=float,
                    help='Define custom test size ratio')
opts = parser.parse_args()

torch.manual_seed(opts.seed) #! important for reproducibility

test_ratio = opts.testratio #used in imagedataset class when defining train test split.

#%% dataset
class imagedataset(Dataset):

    def __init__(self,csv_file,csv_dir,train=True,transform=None): 

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
        
        #define train test splits
        self.X_train, self.X_test, self.y_train, self.y_test =train_test_split(self.X, self.y, test_size=0.33, random_state=123) 
       
        self.X_train = self.X_train.reset_index(drop=True) #drop the indices here or you'll get some exotic errors when loading the files
        #self.y_train = self.y_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        #self.y_test = self.y_test.reset_index(drop=True)
       
        if train==True: #based on argument when defining dataset
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
        image = Image.open(xout) #loads image with PIL
        
        yout = self.yout[index]
        
        if self.transform:
            image = self.transform(image) #apply transformations
            
        return image, yout

def check_data(dataset,verbose = False):
    '''check if there are errors when trying to load the image files and labels'''
    error_list = []
    for ii in range(dataset.len):
        try:
            test = dataset[ii]
        except:
            error_list.append(ii)
            
    error_len = len(error_list)
    data_len = dataset.len
    succes_len = data_len - error_len
      
    if verbose == True:
        print("errors found in the following indices:")
        print(error_list)
        print(dataset.X_train.iloc[error_list])
    else: 
        print(f'succesfully loaded {succes_len} of {data_len} files')
        
def check_image_channels(dataset):
    '''check how many channels there are per image. If there is a difference, solve it '''
    channels_list = []
    for ii in range(dataset.len):
        image = dataset[ii][0]
        channels = transforms.functional.get_image_num_channels(image)
        channels_list.append(channels)
    present_channels = set(channels_list)
    print(f'set of channels in dataset is: {present_channels}')
    
def preview_sample(dataset,sample):
    '''plot a preview of a brain image with corresponding label'''
    fig = plt.imshow(dataset[sample][0].numpy()[0], cmap='gray')
    plt.title(dataset[sample][1])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def check_label_distribution(dataset):
    original_samples = list(dataset.y.numpy())
    label_ratio_original = sum(original_samples) / len(original_samples)
    print(f'ORIGINAL data has a labelratio of: {label_ratio_original}')
    train_samples = list(dataset.y_train.numpy())
    label_ratio_train = sum(train_samples) / len(train_samples)
    print(f'TRAIN data has a labelratio of: {label_ratio_train}')
    test_samples = list(dataset.y_test.numpy())
    label_ratio_test = sum(test_samples) / len(test_samples)
    print(f'VAL data has a labelratio of: {label_ratio_test}')
    
    

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

train_dataset = imagedataset('data.csv', 'C:/Users/rober/Desktop',train=True,transform=image_transforms)
validation_dataset = imagedataset('data.csv', 'C:/Users/rober/Desktop',train=False,transform=image_transforms)
print('quality control')
print('#### training dataset: ####')
check_data(train_dataset)
check_image_channels(train_dataset)
print('#### validation dataset: ####')
check_data(validation_dataset)
check_image_channels(validation_dataset)
print('quality control')
check_label_distribution(train_dataset)

preview_sample(train_dataset,163)

train_loader = DataLoader(dataset= train_dataset, batch_size = 5)
validation_loader = DataLoader(dataset = validation_dataset, batch_size = 5)

                

#%% model
model = models.resnet18(pretrained=True) # loading pretrained resnet model

for param in model.parameters(): #freezing all the layers I don't want to change
    param.requires_grad = False

model.fc = nn.Linear(512,2) #takes 512 inputs out of the convolution, sending it to 2 neurons for me at the end.

criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad],lr=0.003)

#%% Select GPU for training.
use_gpu = torch.cuda.is_available()
if use_gpu:
    print ('GPU Activated')
    model = model.to('cuda')
else:
	print ('CPU is used')


#%% training loop 
max_epochs = 100
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)

for epoch in range(max_epochs):
    loss_sublist = []
    for x,y in train_loader:
        if use_gpu:
            x = x.to('cuda')
            y = y.to('cuda')
        #x,y = x.cuda(),y.cuda() #run on cuda
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss_sublist.append(loss.item())
        loss.backward()
        optimizer.step()
    loss_list.append(np.mean(loss_sublist))
    
    correct=0
    for x_test,y_test in validation_loader:
        if use_gpu:
            x_test = x_test.to('cuda')
            y_test = y_test.to('cuda')
        model.eval()
        z = model(x_test)
        _,yhat = torch.max(z.data,1)
        correct += (yhat==y_test).sum().item()
    accuracy = correct / n_test
    accuracy_list.append(accuracy)
    print(f'==> epoch: {epoch:03d} | Loss: {np.mean(loss_sublist):.4f} | Accuracy: {accuracy:.4f}')

plt.plot(loss_list)
plt.plot(accuracy_list)
plt.legend(['val loss','val accuracy'])
plt.xlabel('epoch')

# checkpoint handling.


