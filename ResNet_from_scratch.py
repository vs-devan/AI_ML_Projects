'''
Implementing ResNet from scratch and training using cifar-10 dataset.
'''

from google.colab import drive
drive.mount('/content/drive')

from IPython.display import clear_output

!pip3 install pyprind

clear_output()

# Downloading and Preparing the Dataset

!gdown --id 1oYnD7Izl3LVVzjEMyLxLklX30TKWHgGG
!unzip /content/cifar-10.zip
!rm -rf /content/cifar-10.zip
!mv /content/cifar-10/sample_submission.csv /content/cifar-10/test_labels.csv

clear_output()

# Imports

import torch
import torchvision

from PIL import Image

import pandas
import numpy
from sklearn import preprocessing
import matplotlib


import os
import pyprind
import cv2
PATH = "/content/cifar-10" 

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        self.entry = pandas.read_csv(os.path.join(self.root_dir, f'{self.mode}_labels.csv'))
        self.encoder = self._process_()
        self.entry['label'] = self.encoder.transform(self.entry['label'])

        self.transform = torchvision.transforms.Compose(
            [
                # Add transforms?
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

    def _process_(self):
        data = pandas.read_csv(os.path.join(self.root_dir, 'train_labels.csv'))
        encoder = preprocessing.LabelEncoder()
        encoder.fit(data['label'])
        return encoder

   
    def __getitem__(self, index):
        train_dir = "/content/cifar-10/train" 
        data = self.entry.iloc[index]
        image = os.path.join(train_dir, str(data['id']))# Read Image
        image=image+".png"
        image = Image.open(image)
        print(image)
        image = self.transform(image)
        label = data['label'] # Read Label?
        return image, label

    def __len__(self):
        return len(self.entry)


    
class Network(torch.torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Network, self).__init__()
        # The first layer of ResNet is a convolutional layer with 64 filters of size 7x7.
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        # The second layer of ResNet is a max pooling layer with kernel size 3x3 and stride 2x2.
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # The third layer of ResNet is a residual block.
        self.resblock1 = self._make_residual_block(64)

        # The fourth layer of ResNet is a residual block.
        self.resblock2 = self._make_residual_block(128)

        # The fifth layer of ResNet is a residual block.
        self.resblock3 = self._make_residual_block(256)

        # The sixth layer of ResNet is a residual block.
        self.resblock4 = self._make_residual_block(512)

        # The seventh layer of ResNet is an average pooling layer with kernel size 7x7 and stride 1x1.
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))

        # The eighth layer of ResNet is a fully connected layer with 10 neurons.
        self.net = torch.nn.Linear(512, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        

        out = self.net(x)
        return out

    def _make_residual_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(channels),
        )


class Trainer():
    def __init__(self, data):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainloader, self.validloader, self.testloader = self.get_iterator(data)
        
        self.model = self.get_model().to(self.device)
        self.criterion = self.get_criterion().to(self.device)
        self.optimizer = self.get_optimizer()

        self.train_loss = []
        self.train_metrics = []
        self.valid_loss = []
        self.valid_metrics = []

        self.epochs = 10

    def get_iterator(self, data):
        train, valid, test = data
        trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True) 
        validloader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
        return trainloader, validloader, testloader

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss() # Add loss
    
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters()) # Add optimizer

    def get_model(self):
        model = Network(num_classes=10).to(self.device) # Add model
        return model

    def save(self, epoch):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(PATH, "model.pth"))
        
    def load(self):
        if os.path.exists(os.path.join(PATH, "model.pth")):
            checkpoints = torch.load(os.path.join(self.args.checkpoint, "model.pth"), map_location=self.device)
            self.model.load_state_dict(checkpoints['model_state_dict'])
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

    def train(self):
        epoch_loss = 0
        epoch_metrics = {}

        self.model.train()

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.trainloader), bar_char=' ')
            for index, (image, label) in enumerate(self.trainloader):
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                
                output = self.model(image) # Forward Pass

                loss = self.criterion(output, label) # Evaluate Loss

                loss.backward() # Back Propogation

                self.optimizer.step()
                bar.update()

        return epoch_loss, epoch_metrics

    def evaluate(self):
        epoch_loss = 0
        epoch_metrics = {}

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.validloader), bar_char='█')
            for index, (image, label) in enumerate(self.validloader):
                image = image.to(self.device)
                label = label.to(self.device)
                
                output = self.model(image) # Forward Pass

                loss = self.criterion(output, label) # Evaluate Loss

                bar.update()

        return epoch_loss, epoch_metrics

    def test(self):

        self.model.eval()

        outputs = torch.empty([0,])

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.testloader), bar_char='█')
            for index, (image, label) in enumerate(self.testloader):
                image = image.to(self.device)
                label = label.to(self.device)
                
                output = self.model(image).detach().cpu()
                outputs = torch.cat((outputs, output), dim=0)

                bar.update()

        return outputs
    
    def fit(self):

        for epoch in range(1, self.epochs+1, 1):

            epoch_train_loss, epoch_train_metrics = self.train()

            self.train_loss.append(epoch_train_loss)
            self.train_metrics.append(epoch_train_metrics)

            epoch_valid_loss, epoch_valid_metrics = self.evaluate()
            
            self.valid_loss.append(epoch_valid_loss)
            self.valid_metrics.append(epoch_valid_metrics) 

            print(f'Epoch {epoch}/{self.epochs+1}: Train Loss = {epoch_train_loss} | Validation Loss = {epoch_valid_loss}')

            if epoch_valid_metrics >= max(self.valid_metrics):
                self.save(epoch)




train_data = CreateDataset(root_dir="/content/cifar-10") 
train_data, valid_data = torch.utils.data.random_split(train_data, [len(train_data)-len(train_data)//10, len(train_data)//10])
test_data = CreateDataset(root_dir="/content/cifar-10")
data = (train_data, valid_data, test_data)

trainer = Trainer(data)
trainer.fit()

outputs = trainer.test()
