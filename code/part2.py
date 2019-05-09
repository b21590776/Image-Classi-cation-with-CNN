import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.legend_handler import HandlerLine2D
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


# parameters
learning_rate = 0.001
epochs = 10
batch_size = 25
num_workers = 2

# class names
classes = ['airport_inside', 'bar', 'bedroom', 'casino', 'inside_subway', 'kitchen',
           'livingroom', 'restaurant', 'subway', 'warehouse']

train_on_gpu = torch.cuda.is_available()
data_dir = "dataset"
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]),
        'validation': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]),
    }
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'validation', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
               for x in ['train', 'validation', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}


# loading the pretrained model from pytorch
model_ft = models.vgg16(pretrained=True)

# model structure
# print(vgg16)

# freezing training for all "features" layers
for param in model_ft.features.parameters():
    param.requires_grad = False


features = model_ft.classifier[0].in_features
FC = nn.Linear(features, 2048)
FC_OUT = nn.ReLU(FC)
FC1 = nn.Linear(2048, 2048)
FC1_OUT = nn.ReLU(FC1)
FC2 = nn.Linear(2048, len(classes))
FC_LAYERS = [FC, FC_OUT, FC1, FC1_OUT, FC2]

# replacing the model classifier
model_ft.classifier = nn.Sequential(*FC_LAYERS).cuda()

# if GPU is available, move the model to GPU
if train_on_gpu:
    model_ft.cuda()

# adam optimizer
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=learning_rate)

# specify loss function cross-entropy
criterion = nn.CrossEntropyLoss()

# bonus
# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(epoch, model, train_loader, optimizer):
    correct_train = 0
    train_loss = 0.0
    temp = 0.0
    # model is setting to train
    for batch_i, (data, target) in enumerate(train_loader):
        data = data.cuda()
        labels = target.cuda()

        # moving tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clearing the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: computing predicted outputs by passing inputs to the model
        output = model(data)
        # calculating the batch loss
        loss = criterion(output, target)
        # backward pass: computing gradient of the loss with respect to model parameters
        loss.backward()
        # performing a single optimization step (parameter update)
        optimizer.step()
        # updating training loss
        train_loss += loss.item()
        temp += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_train += pred.eq(labels.data.view_as(pred)).cpu().sum()
        loss = temp / batch_size

        # printing training loss every specified number of mini-batches
        if batch_i % batch_size == batch_size-1:
            print('Epoch %d, Batch %d, Loss: %.7f' %(epoch, batch_i + 1, loss))
            temp = 0.0

    train_loss /= len(train_loader)

    print('\nTrain set = Epoch: {}\tAccuracy {}/{} ({:.0f}%)\tAverage loss: {:.7f}\n'.format(
        epoch, correct_train, len(train_loader) * batch_size,
                              100. * correct_train / (len(train_loader) * batch_size), train_loss))

    acc = '{:.0f}'.format(100. * correct_train / (len(train_loader) * batch_size))

    return train_loss, float(acc)/100


def validation(epoch, model, valid_loader):
    model.eval()
    valid_loss = 0
    correct_valid = 0
    for data, label in valid_loader:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        valid_loss += criterion(output, label).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader)

    print('\nValidation set = Epoch: {}\tAccuracy: {}/{} ({:.0f}%)\tAverage loss: {:.7f}\n'.format(
        epoch, correct_valid, (len(valid_loader) * batch_size),
        100. * correct_valid / (len(valid_loader) * batch_size), valid_loss))

    acc = '{:0.0f}'.format(100. * correct_valid / (len(valid_loader)* batch_size))
    return valid_loss, float(acc)/100


def test(model, loader):

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_correct5 = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_total5 = list(0. for i in range(10))
    model.eval()  # evaluation mode

    # iterating over test data
    for data, target in loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: computing predicted outputs by passing inputs to the model
        output = model(data)
        # calculating the batch loss
        loss = criterion(output, target)
        # updating test loss
        test_loss += loss.item() * data.size(0)
        # converting output probabilities to predicted class
        # Top-1
        _, pred = torch.max(output, 1)

        # Top-5
        _5, pred5 = torch.topk(output, 5)

        # calculating test accuracy for Top-5
        for i in range(len(pred5)):
            label = target.data[i]
            class_correct5[label] += int(torch.sum(label == pred5[i]))
            class_total5[label] += 1

        # comparing predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculating test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculating average test loss
    test_loss = test_loss / len(loader.dataset)
    print('Test Loss: {:.7f}\n'.format(test_loss))

    # Top 1 and Top 5 accuracy of the test set
    for i in range(len(classes)):
        print('Test Accuracy of %5s: Top-1 %2d%% (%2d/%2d), Top-5 %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i]),
            100 * class_correct5[i] / class_total5[i], np.sum(class_correct5[i]), np.sum(class_total5[i])
        ))

    print('\n Overall Test Accuracy : Top-1 %2d%% (%2d/%2d), Top-5 %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total),
        100. * np.sum(class_correct5) / np.sum(class_total5), np.sum(class_correct5), np.sum(class_total5)
    ))


def train_model(model, train_loader, valid_loader, test_loader, epochs):
    x = list()
    train_y = list()
    valid_y = list()
    train_a = list()
    valid_a = list()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(epoch, model, train_loader, optimizer)
        valid_loss, valid_acc = validation(epoch, model, valid_loader)
        x.append(epoch)
        train_a.append(train_acc)
        valid_a.append(valid_acc)
        train_y.append(train_loss)
        valid_y.append(valid_loss)

    test(model, test_loader)
    plot_loss_table(x, train_y, valid_y)
    plot_accuracy_table(x, train_a, valid_a)

    # saving the model
    torch.save(model.state_dict(), os.path.curdir + "part2.pt")


def plot_accuracy_table(x, train_y, valid_y):
    fig = plt.figure(0)
    fig.canvas.set_window_title('Train accuracy vs Validation accuracy')
    plt.axis([0, epochs + 1, 0, 1])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    train_graph, = plt.plot(x, train_y, 'b--', label='Train accuracy')
    plt.plot(x, valid_y, 'g', label='Validation accuracy')
    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


def plot_loss_table(x, train_y, valid_y):

    fig = plt.figure(0)
    fig.canvas.set_window_title('Train loss vs Validation loss')
    plt.axis([0, epochs + 1, 0, 2.5])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    train_graph, = plt.plot(x, train_y, 'b--', label='Train loss')
    plt.plot(x, valid_y, 'g', label='Validation loss')
    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


if __name__ == '__main__':
    train_model(model_ft, dataloaders['train'], dataloaders['validation'], dataloaders['test'], epochs)

