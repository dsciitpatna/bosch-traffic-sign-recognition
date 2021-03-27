from .models import CSV
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect

import csv
import io
import logging
logger = logging.getLogger(__name__)

# Upload of csv file

import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from torchsummary import summary
from scipy.ndimage.filters import convolve
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import pandas as pd
import os

import warnings

#ML imports


def upload_file(req):
    template_name = "upload_csv.html"
    data = CSV.objects.all()
    context = {
        'order': 'Order of the CSV should be Width,Height,Ro1.x1,Roi.y1,Roi.x2,Roi.y2,ClassId,Path',
    }
    if req.method == "GET":
        return render(req, template_name, context)
    try:
        csv_file = req.FILES['file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect('csvFile:upload_file')
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (
                csv_file.size/(1000*1000),))
            return redirect('csvFile:upload_file')

        data_set = csv_file.read().decode('UTF-8')

        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = CSV.objects.update_or_create(
                Width=column[0],
                Height=column[1],
                Roi_x1=column[2],
                Roi_y1=column[3],
                Roi_x2=column[4],
                Roi_y2=column[5],
                ClassId=column[6],
                path=column[7],
                Class_name=column[8])
        context['success'] = "Succesfully loaded the data"
    except:
        logger.error(req, "oject not found")
        context['failure'] = "Failed to load the data"
    return render(req, template_name, context)


# Export csv file
def export_file(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="users.csv"'

    writer = csv.writer(response)
    writer.writerow(['Width', 'Height', 'Roi_x1', 'Roi_y1', 'Roi_x2','Roi_y2','ClassId','path','Class_name'])

    data = CSV.objects.all().values_list(
       'Width', 'Height', 'Roi_x1', 'Roi_y1', 'Roi_x2','Roi_y2','ClassId','path','Class_name')
    for d in data:
        writer.writerow(d)

    return response

# displaying data


def show(req):
    template_name = "show.html"
    data = CSV.objects.all()
    context = {"open": "Page opened", "data": data}
    return render(req, template_name, context)

def train_model(req):
    template_name = "train.html"
    train()
    return render(req,template_name)



def train():
    warnings.filterwarnings("ignore")

    epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE=50

    train_dir = './Train'      ## path to train dataset
    test_dir = '../Test'            ## path to test dataset
    test_csv = './users.csv'    ## path to test.csv
    save_dir = './'                                             ## path to directory to store trained models

    ## for creating test dataset
    class test_dataset(Dataset):

        def __init__(self, test_dir, csv_file, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.data = pd.DataFrame(list(CSV.objects.all().values()))
            self.root_dir = test_dir
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.root_dir,self.data.iloc[idx, -1])
            image = io.imread(img_name)
            true_lab = self.data.iloc[idx, -2]
            
            if self.transform:
                image = self.transform(image)
                
            sample = (image,true_lab)

            return sample
        

    class LocalContrastNormalization(object):

        def __init__(self, kernel_size=3, mode='constant', cval=0.0):
            self.kernel_size = kernel_size
            self.mode = mode
            self.cval = cval
            
        def __call__(self, tensor):
        
            return torch.stack([self.func(torch.tensor(batch)) for batch in tensor.tolist()])
            

        def func(self, tensor):
        
            C, H, W = tensor.size()
            kernel = np.ones((self.kernel_size, self.kernel_size))
            

            arr = np.array(tensor)
            local_sum_arr = np.array([convolve(arr[c], kernel, mode=self.mode, cval=self.cval)
                                    for c in range(C)]) # An array that has shape(C, H, W)
                                                        # Each element [c, h, w] is the summation of the values
                                                        # in the window that has arr[c,h,w] at the center.
            local_avg_arr = local_sum_arr / (self.kernel_size**2) # The tensor of local averages.

            arr_square = np.square(arr)
            local_sum_arr_square = np.array([convolve(arr_square[c], kernel, mode=self.mode, cval=self.cval)
                                    for c in range(C)]) # An array that has shape(C, H, W)
                                                        # Each element [c, h, w] is the summation of the values
                                                        # in the window that has arr_square[c,h,w] at the center.
            local_norm_arr = np.sqrt(local_sum_arr_square) # The tensor of local Euclidean norms.


            local_avg_divided_by_norm = local_avg_arr / (1e-8+local_norm_arr)

            result_arr = np.minimum(local_avg_arr, local_avg_divided_by_norm)
            
            return torch.Tensor(result_arr)



        def __repr__(self):
            return self._class.name_ + '(kernel_size={0}, threshold={1})'.format(self.kernel_size, self.threshold)
        
        
    ## Model architecture defination

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 200, kernel_size=7, padding=2)
            self.max1 = nn.MaxPool2d(2, stride=2)
            self.conv2 = nn.Conv2d(200, 250, kernel_size=4, padding=2)
            self.max2 = nn.MaxPool2d(2, stride=2)
            self.conv3 = nn.Conv2d(250, 350, kernel_size=4, padding=2)
            self.max3 = nn.MaxPool2d(2, stride=2)
            self.local = LocalContrastNormalization()
            self.conv_drop = nn.Dropout2d(p=0.5)
            self.fc1 = nn.Linear(350*6*6, 400)
            self.fc2 = nn.Linear(400, 43)

            # stn1 localizaton net
            self.localization1 = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(3, 250, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(250, 250, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc1 = nn.Sequential(
                nn.Linear(250 * 6 * 6, 250),
                torch.nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Linear(250, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc1[3].weight.data.zero_()
            self.fc_loc1[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
            # stn2 localizaton net
            self.localization2 = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(200, 150, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(150, 200, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc2 = nn.Sequential(
                nn.Linear(200 * 2 * 2, 300),
                torch.nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Linear(300, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc2[3].weight.data.zero_()
            self.fc_loc2[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
            # stn3 localizaton net
            self.localization3 = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(250, 150, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(150, 200, kernel_size=5, padding=2),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc3 = nn.Sequential(
                nn.Linear(200 * 1 * 1, 300),
                torch.nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Linear(300, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc3[3].weight.data.zero_()
            self.fc_loc3[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # stn1
        def stn1(self, x):
            xs1 = self.localization1(x)
            xs1 = xs1.view(-1, 250 * 6 * 6)
            theta1 = self.fc_loc1(xs1)
            theta1 = theta1.view(-1, 2, 3)

            grid1 = F.affine_grid(theta1, x.size())
            x1 = F.grid_sample(x, grid1)

            return x1
        
        # stn2
        def stn2(self, x):
            xs2 = self.localization2(x)
            xs2 = xs2.view(-1, 200 * 2 * 2)
            theta2 = self.fc_loc2(xs2)
            theta2 = theta2.view(-1, 2, 3)

            grid2 = F.affine_grid(theta2, x.size())
            x2 = F.grid_sample(x, grid2)

            return x2
        
        # stn3
        def stn3(self, x):
            xs3 = self.localization3(x)
            xs3 = xs3.view(-1, 200 * 1 * 1)
            theta3 = self.fc_loc3(xs3)
            theta3 = theta3.view(-1, 2, 3)

            grid3 = F.affine_grid(theta3, x.size())
            x3 = F.grid_sample(x, grid3)

            return x3

        def forward(self, x):
            # transform the input
            x = self.stn1(x)
            x = self.conv_drop(F.relu(self.conv1(x)))
            x = self.max1(x)
            x = self.local(x).to(device)
            
            x = self.stn2(x)
            x = self.conv_drop(F.relu(self.conv2(x)))
            x = self.max2(x)
            x = self.local(x).to(device)
            
            x = self.stn3(x)
            x = self.conv_drop(F.relu(self.conv3(x)))
            x = self.max3(x)
            x = self.local(x).to(device)
            
            x = x.view(-1, 350*6*6)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

    ## transforms for train data
    trans = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1,1,1])
    ])

    ## transforms for test data

    test_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1,1,1])
    ])

    tr_data = datasets.ImageFolder(train_dir,transform = trans)
    train_data_loader = data.DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = test_dataset(test_dir,test_csv,transform = test_trans)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    def train(dataloader, model, criterion, optimizer):
        size = len(dataloader.dataset)
        preds=[]
        true=[]
        tot_loss=0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            tot_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            z = F.softmax(pred,dim=1)
            z = torch.argmax(z,dim=1)
            preds += z.tolist()
            true += y.tolist()

            if batch % 100 == 0:
                print('batch : {} = loss : {}'.format(batch+1,loss.item()))
        
        accuracy = accuracy_score(true,preds)
        tot_loss = tot_loss/(batch+1)     
        return (tot_loss,accuracy)

    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        preds=[]
        true=[]
        with torch.no_grad():
            for batch,data in enumerate(test_data_loader):
                X = data[0].to(device)
                y = data[1].to(device)
                pred = model(X)
                test_loss += criterion(pred, y).item()
                z = F.softmax(pred,dim=1)
                z = torch.argmax(z,dim=1)
                preds += z.tolist()
                true += y.tolist()
                
        test_accuracy = accuracy_score(true,preds)
        test_loss /= (batch+1)
        return (test_loss,test_accuracy)

    ## Training......................

    epoch_loss=[]
    acc_list=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss,train_accuracy = train(train_data_loader, model, criterion, optimizer)
        
        print('accuracy : {}'.format(100*train_accuracy))
        print('epoch loss : {}'.format(train_loss))
        
        test_loss,test_accuracy = test(test_data_loader, model)
        epoch_loss.append({'train loss':train_loss,'test loss':test_loss})
        acc_list.append({'train acc':train_accuracy,'test acc':test_accuracy})
        
        print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}, Avg loss: {test_loss:>8f} \n")
        torch.save(model, os.path.join(save_dir,f'model_ep{t+1}.pt'))

    print("Training Done!")
