import os
import pandas as pd

# define folder structure
dir_path = 'C:/Users/rober/Desktop/'
dir_list = ['yes','no'] #add folder names in this list

dir_labels = ['tumor','normal'] # add labels you want into this list (in order!)
df = pd.DataFrame(columns = ['image_path','labels'])
for ii in range(len(dir_list)):
    #navigate to the dirs and select corresponding label
    fulldirpath = os.path.join(dir_path,dir_list[ii])
    os.chdir(fulldirpath)
    label = dir_labels[ii]
    #make a dataframe to but the next files into
    
    file_list = []
    
    for file in os.listdir():
        full_file_path = os.path.join(fulldirpath,file).replace("\\","/")
        # write individual files to a csv, with corresponding labels
        file_list.append(full_file_path)
        
        labels = [label]*len(file_list)
        df1 = pd.DataFrame([file_list,labels]).T
        df1.columns = ['image_path','labels']

    df = pd.concat([df,df1])
        
        
os.chdir(dir_path)
df.to_csv('data.csv',index=False)     
        
        
    

