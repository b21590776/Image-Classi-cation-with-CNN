import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# parameters
batch_size = 25
num_workers = 4

lin_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                        verbose=0)


def train(train_feature, label):
    lin_clf.fit(train_feature, label)


def predict(vec):
    return lin_clf.predict(vec)


data_dir = "dataset"

# class names
classes = ['airport_inside', 'bar', 'bedroom', 'casino', 'inside_subway', 'kitchen',
           'livingroom', 'restaurant', 'subway', 'warehouse']

train_on_gpu = torch.cuda.is_available()


def main():

    # data transforms
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
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
                   for x in ['train', 'validation', 'test']}

    model = models.vgg16(pretrained=False)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[0].in_features
    FC = nn.Linear(features, 2048)
    FC_OUT = nn.ReLU(FC)
    FC1 = nn.Linear(2048, 2048)
    FC1_OUT = nn.ReLU(FC1)
    FC2 = nn.Linear(2048, len(classes))
    FC_LAYERS = [FC, FC_OUT, FC1, FC1_OUT, FC2]

    # replacing the model classifier
    model.classifier = nn.Sequential(*FC_LAYERS).cuda()

    # Loading the pretrained model from part2
    model.load_state_dict(torch.load(os.path.curdir + "part2.pt"))
    # removing last layer
    features = list(model.classifier.children())[:-2]
    # replacing the model classifier
    model.classifier = nn.Sequential(*features)

    if train_on_gpu:
        model.cuda()

    model.eval()

    # train
    feature_train = []
    label_train = []
    with torch.no_grad():
        for img, label in dataloaders['train']:
            if train_on_gpu:
                img, label = Variable(img.cuda()), Variable(label.cuda())
            else:
                img, label = Variable(img), Variable(label)

            feature_train.append(model(img))
            label_train.append(label)
    feature_train = torch.cat(feature_train, dim=0)
    feature_train = feature_train.detach().cpu().clone().numpy()
    label_train = torch.cat(label_train, dim=0)
    label_train = label_train.detach().cpu().clone().numpy()
    # test
    feature_test = []
    label_test = []
    with torch.no_grad():
        for img, label in dataloaders['test']:
            if train_on_gpu:
                img, label = Variable(img.cuda()), Variable(label.cuda())
            else:
                img, label = Variable(img), Variable(label)

            feature_test.append(model(img))
            label_test.append(label)
    feature_test = torch.cat(feature_test, dim=0)
    feature_test = feature_test.detach().cpu().clone().numpy()
    label_test = torch.cat(label_test, dim=0)
    label_test = label_test.detach().cpu().clone().numpy()

    train(feature_train, label_train)
    pred_test = predict(feature_test)
    overall_correct = 0
    class_correct = 0
    for i in range(len(pred_test)):
        if pred_test[i] == label_test[i]:
            class_correct += 1
            overall_correct += 1
        if (i + 1) % 25 == 0:
            print("Test Accuracy of Class :", classes[i // 25] , ", ", class_correct / 25)
            class_correct = 0

    print("Overall Test Accuracy :", overall_correct / len(pred_test))

    classs = np.array(['airport_inside', 'bar', 'bedroom', 'casino', 'inside_subway', 'kitchen', 'livingroom',
                       'restaurant', 'subway', 'warehouse'])

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(label_test, pred_test, classes=classs,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(label_test, pred_test, classes=classs, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


# confusion matrix function ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix
# .htmlsphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    main()
