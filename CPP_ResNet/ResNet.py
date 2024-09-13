import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm
import random
import time


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet(BasicBlock, [2, 2, 2, 2])
net = net.to(device)
# summary(net, input_size=(3, 32, 32))

# Move dataset loading outside of the main function so it is only executed once
train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
test_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], 
                                                std = [0.2023, 0.1994, 0.2010])
                       ])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

def train(data_loader, model, criterion, optimizer, scheduler=None, early_stop=None):
    learning_rate_tracker = {}
    epoch_correct = 0
    running_loss = 0.0
    model.train()
    
    # Wrapping the enumerator with tqdm for progress bar
    for i, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
        learning_rate_tracker[i] = optimizer.param_groups[0]['lr']
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) 
        running_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        epoch_correct += (predicted == labels).sum().item()
        
        if early_stop and i == early_stop:
            break
            
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    return epoch_correct, running_loss, learning_rate_tracker
    
def evaluate(data_loader, model, criterion):
    epoch_correct = 0
    running_loss = 0.0
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) 
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return epoch_correct, running_loss, y_true, y_pred

def main():
    training_start = time.time()
    lr_min = 1e-6
    lr_max = 1e-2
    epochs = 30
    step_size = (len(train_dataset)/64) // 2

    model = net
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_min, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=step_size, step_size_down=step_size, gamma=0.9999, mode="exp_range", cycle_momentum=False)
    lr_tracker = {}

    train_loss_history_cuda = []
    train_acc_history_cuda = []
    val_loss_history_cuda = []
    val_acc_history_cuda = []
    y_pred_cuda = []
    y_true_cuda = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")
        train_start = time.time()
        correct, loss, rate_tracker = train(data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        train_end = time.time()
        print(f"Train time: {train_end - train_start} seconds")
        accuracy = correct / len(train_loader.dataset)
        loss = loss / len(train_loader)
        train_loss = loss
        train_acc_history_cuda.append(accuracy)
        train_loss_history_cuda.append(loss)
        for key in rate_tracker.keys():
            lr_tracker[(epoch,key)] = rate_tracker[key]
        
        eval_start = time.time()
        correct, loss, y_true_cuda, y_pred_cuda = evaluate(data_loader = test_loader, model=model, criterion=criterion)
        eval_end = time.time()
        print(f"Evaluation time: {eval_end - eval_start} seconds")
        validation_accuracy = correct / len(test_loader.dataset)
        validation_loss = loss / len(test_loader)
        print(f"Train Accuracy: {accuracy*100:.2f}%, Train Loss: {train_loss}")
        print(f"Test Accuracy: {validation_accuracy*100:.2f}%, Test Loss: {validation_loss}")
        # if validation_loss < best_valid_loss:
        #     best_valid_loss = validation_loss
        #     torch.save(model.state_dict(), 'ResNetCuda.pt')
        val_acc_history_cuda.append(validation_accuracy)
        val_loss_history_cuda.append(validation_loss)
        epoch_end = time.time()
        print(f"Epoch time: {epoch_end - epoch_start} seconds") 
    
    training_end = time.time()
    print(f"Total training time: {training_end - training_start} seconds")
        
if __name__ == "__main__":
    main()