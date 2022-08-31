import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    if training:
        data_out = torch.utils.data.DataLoader(train_set, batch_size=64)
    else:
        data_out = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=64)
    return data_out

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                          nn.Linear(64, 10))
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        total = 0
        correct = 0
        flag = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            flag += 1
        percent = "{:.2%}".format(correct / total)
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({percent}) Loss: {round(running_loss / flag, 3)}")


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    flag = 0
    with torch.no_grad():
        for data, labels in test_loader:
            flag += 1
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        if show_loss:
            print(f"Average loss: {round(running_loss/flag,4)}")
        print('Accuracy: ' + '{0:.2f}'.format(100 * correct / total) + '%')


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']
    prob = F.softmax(model(test_images[index]).flatten(), dim=0)
    sort_list = list(torch.argsort(prob))[::-1]
    print("{:s}: {:.2%}".format(class_names[sort_list[0]], prob[sort_list[0]]))
    print("{:s}: {:.2%}".format(class_names[sort_list[1]], prob[sort_list[1]]))
    print("{:s}: {:.2%}".format(class_names[sort_list[2]], prob[sort_list[2]]))

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    #train_loader = get_data_loader()
    #print(type(train_loader))
    #print(train_loader.dataset)
    #test_loader = get_data_loader(False)
    #model = build_model()
    #print(model)
    #train_model(model, train_loader, criterion, T=5)
    #evaluate_model(model, test_loader, criterion, show_loss = False)
    #evaluate_model(model, test_loader, criterion, show_loss = True)
    #tester = next(iter(test_loader))[0]
    #predict_label(model, tester, 1)
