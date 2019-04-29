import numpy as np
from tqdm import tqdm
from random import shuffle
import os, math, cv2, random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
 

'''
Helper function to deal with image data; loading, plotting etc

'''    
    
    
IMG_SIZE = 299  #Fixed 
img_shape = (IMG_SIZE,IMG_SIZE,3)  

def load_data(num_data = 500, IMG_SIZE = 100, DIR = os.getcwd()):
    data = []
    labels = []
    for directory in os.listdir(DIR):
        if not directory.startswith('.'):
            path1 = os.path.join(DIR, directory)
            for i, img in tqdm(enumerate(os.listdir(path1))):
            	if (i < num_data):
            		if img.startswith('.'):
            			num_data += 1
            		if not img.startswith('.'):
                		path2 = os.path.join(path1, img)
                		img = cv2.imread(path2)
                		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                		label = directory
                		labels.append([np.array(label)])
                		data.append([np.array(img)])
    #print("{0} Files loaded from each classes".format(i+1))
    random.Random(1).shuffle(data)
    random.Random(1).shuffle(labels)
    return np.array(data).reshape(-1,IMG_SIZE,IMG_SIZE,3), np.array(labels)


def plot_images(images, cls_true, cls_pred = None):
    assert len(images) == len(cls_true) == 10 

    fig, axes = plt.subplots(2,5, figsize = (15,5))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap = 'binary')
        
        if cls_pred is None:
            xlabel = "True: {}".format(cls_true[i])
        else:
            xlabel = "True: {0}, \n Pred = {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel, fontsize = 12)
        
        #Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()


def plot_confusion_matrix(cls_true, cls_pred, labels):
    
     
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred, labels = labels)
    #print(cm)

    
    #plt.matshow(cm)
    #fig = plt.figure()
    ax = sns.heatmap(cm, annot=True, cmap='YlGnBu', linewidths=.5, linecolor='white')

    # Make various adjustments to the plot.
    plt.title('Confusion matrix of the classifier')

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted', fontsize = 12)
    plt.ylabel('True', fontsize = 12)
    ax.figure.set_size_inches((12, 10))
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    
    #print(cm)
    plt.show()

def misclassified_images(true_cls, pred_cls):
    misclassified = []
    for i in range(len(true_cls)):
        if true_cls[i] != pred_cls[i]:
            misclassified.append(i)
    return misclassified


def plot_images_misclassified(images, true_cls, pred_cls, miss): #Plot misclassified images
    fig, axes = plt.subplots(5,5, figsize = (15,15))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap = 'binary')
        xlabel = "True : {0}, \n Pred :{1} \n index : {2}".format(true_cls[i], pred_cls[i], miss[i])
        ax.set_xlabel(xlabel, fontsize = 12)
        
        #Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()


def random_mini_batch(X, Y, minibatch_size = 8): #Old function; Not efficient
    minibatches = []
    m = X.shape[0]
    #Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    
    num_complete_minibatches = math.floor(m/minibatch_size)
    for i in range(num_complete_minibatches):
        minibatch_x = shuffled_X[(i*minibatch_size):((i+1)*minibatch_size),:,:,:]
        minibatch_y = shuffled_Y[(i*minibatch_size):((i+1)*minibatch_size),:]
        minibatch = (minibatch_x,minibatch_y)
        minibatches.append(minibatch)
    if m % minibatch_size != 0:
        minibatch_x = shuffled_X[(num_complete_minibatches*minibatch_size):m,:,:,:]
        minibatch_y = shuffled_Y[(num_complete_minibatches*minibatch_size):m,:]
        minibatch = (minibatch_x,minibatch_y)
        minibatches.append(minibatch)
    yield minibatches

def iterate_minibatches(inputs, targets, batchsize): #Using python generator
    assert inputs.shape[0] == targets.shape[0]
    m = inputs.shape[0] 
    indices = np.arange(m)
    np.random.shuffle(indices)
    for index in range(0, m - batchsize + 1, batchsize): # 1 is when SGD
        batch = indices[index:index + batchsize]
        yield inputs[batch], targets[batch]
    if m % batchsize != 0:
        batch = indices[math.floor(m/batchsize)*batchsize:m]
        yield inputs[batch], targets[batch]























