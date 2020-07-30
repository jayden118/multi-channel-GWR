"""
gwr-tb :: utilities
@author: Weihong Chin (weihong118118@gmail.com)

"""

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
#import matplotlib.animation as animation

def import_network(file_name, NetworkClass):
    """ Import pickled network from file
    """
    file = open(file_name, 'br')
    data_pickle = file.read()
    file.close()
    net = NetworkClass()
    net.__dict__ = pickle.loads(data_pickle)
    return net
    
def export_network(file_name, net) -> None:
    """ Export pickled network to file
    """
    file = open(file_name, 'wb')
    file.write(pickle.dumps(net.__dict__))
    file.close()

def load_file(file_name) -> np.ndarray:
    """ Load dataset from file
    """
    reader = csv.reader(open(file_name, "r"), delimiter=',')
    x_rdr = list(reader)
    return  np.array(x_rdr).astype('float')

def normalize_data(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, data.shape[1]):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / (max_col - min_col)
    return data

def normalize_data2(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, data.shape[1]):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / ((max_col - min_col) + 0.0000001)
    return data

def normalize_data3(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, 3):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / ((max_col - min_col) + 0.0000001)
    return data

def normalize_data4(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, 3):
        for j in range(0, data.shape[0]):
            data[j, i] = data[j, i]  / 100
            data[j, i+3] = data[j, i+3] / 10
    return data


def plot_data(ds_vectors, ds_labels, steps) -> None:
    for iteration in range(0, ds_vectors.shape[0]):
        label = ds_labels[0][iteration]
        if label == 2:
            plt.scatter(ds_vectors[iteration][0], ds_vectors[iteration][1], color='red', alpha=.5)
        elif label == 1:
            plt.scatter(ds_vectors[iteration][0], ds_vectors[iteration][1], color='blue', alpha=.5)
        elif label == 0:
            plt.scatter(ds_vectors[iteration][0], ds_vectors[iteration][1], color='black', alpha=.5)
                
    if(steps < 10):
        plt.savefig('dataPlot/000{y}.png'.format(y=steps))
    elif(steps < 100):
        plt.savefig('dataPlot/00{y}.png'.format(y=steps))
    else:
        plt.savefig('dataPlot/0{y}.png'.format(y=steps))
    
    
def animate_network(net, edges, labels, steps)-> None:
    plt.clf()
    plt.title('Iteration {y}'.format(y=steps))
    plt.xlim(-10.0, 10.0)
    plt.ylim(-5.0, 5.0)
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        for l in range(0, len(net.num_labels)):
            plindex = np.argmax(net.alabels[l][ni])
        if labels:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][0], net.weights[nj][0]], 
                                 [net.weights[ni][1], net.weights[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]], 
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3) 
    if(steps < 10):
        plt.savefig('snaps/000{y}.png'.format(y=steps))
    elif(steps < 100):
        plt.savefig('snaps/00{y}.png'.format(y=steps))
    else:
        plt.savefig('snaps/0{y}.png'.format(y=steps))
 
       
def animate_networkPCA(data, net, edges, labels, steps)-> None:
    plt.clf()
    plt.title('Iteration {y}'.format(y=steps))
    plt.xlim(-5.0, 9.0)
    plt.ylim(-5.0, 5.0)
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    dim_net = True if len(data.shape) <= 2 else False
    for ni in range(len(net.weights)):
        for l in range(0, len(net.num_labels)):
            plindex = np.argmax(net.alabels[l][ni])
        if labels:
            if dim_net:
                plt.scatter(data[ni][0], data[ni][1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (data[ni][0], data[ni][1]))
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (net.weights[ni][0, 0], net.weights[ni][0, 1]))
        else:
            if dim_net:
                plt.scatter(data[ni][0], data[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([data[ni][0], data[nj][0]], 
                                 [data[ni][1], data[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]], 
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3) 
    if(steps < 10):
        plt.savefig('snaps/000{y}.png'.format(y=steps))
    elif(steps < 100):
        plt.savefig('snaps/00{y}.png'.format(y=steps))
    else:
        plt.savefig('snaps/0{y}.png'.format(y=steps))
            
    
def plot_network(net, edges, labels) -> None:
    """ 2D plot
    """        
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['blue','black','red','green','yellow','cyan','magenta','0.75','0.15','1']
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        for l in range(0, len(net.num_labels)):
            plindex = np.argmax(net.alabels[l][ni])
        if labels:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][0], net.weights[nj][0]], 
                                 [net.weights[ni][1], net.weights[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]], 
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3)                        
    plt.show()

def plot_networkPCA(data, net, edges, labels) -> None:
    """ 2D plot
        """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','orange','purple']
    plt.figure()
    dim_net = True if len(data.shape) <= 2 else False
    for ni in range(len(net.weights)):
        for l in range(0, len(net.num_labels)):
            plindex = np.argmax(net.alabels[l][ni])
        if labels:
            if dim_net:
                plt.scatter(data[ni][0], data[ni][1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (data[ni][0], data[ni][1]))
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (net.weights[ni][0, 0], net.weights[ni][0, 1]))
        else:
            if dim_net:
                plt.scatter(data[ni][0], data[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([data[ni][0], data[nj][0]],
                                 [data[ni][1], data[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]],
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3)
    #plt.legend(loc='best')
    plt.show()


def plot_network_annotate(net, edges, labels) -> None:
    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        for l in range(0, len(net.num_labels)):
            plindex = np.argmax(net.alabels[l][ni])
        if labels:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (net.weights[ni][0], net.weights[ni][1]))
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=ccc[plindex], alpha=.5)
                plt.annotate(plindex, (net.weights[ni][0, 0], net.weights[ni][0, 1]))
        else:
            if dim_net:
                plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][0], net.weights[nj][0]],
                                 [net.weights[ni][1], net.weights[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]],
                                 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
                                 'gray', alpha=.3)
    plt.show()
    

def PCA_network(net):
    dataBuf = []
    for i in range (0, net.num_nodes):
        if i == 0:
            dataT = np.array([net.weights[i][0]])
            dataBuf = dataT
        else:
            dataT = np.array([net.weights[i][0]])
            dataBuf = np.concatenate((dataBuf, dataT), axis=0)

    pca = PCA(n_components=2)
    dataPCA = pca.fit_transform(dataBuf)
    return dataPCA
    

class IrisDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self, file, normalize, numClass):
        self.name = 'IRIS'
        self.file = file
        self.normalize = normalize
        self.num_classes = numClass # iris class=3 
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data(self.vectors)
            
class CustomDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self, file, normalize, numClass):
        self.name = 'IRIS'
        self.file = file
        self.normalize = normalize
        self.num_classes = numClass # iris class=3 
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data2(self.vectors)

class NCLTodoDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self, file, normalize, numClass):
        self.name = 'NCLT'
        self.file = file
        self.normalize = normalize
        self.num_classes = numClass # iris class=3 
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 1:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data4(self.vectors)
            
class NCLTlaserDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self, file, normalize, numClass):
        self.name = 'NCLT'
        self.file = file
        self.normalize = normalize
        self.num_classes = numClass # iris class=3 
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data2(self.vectors)
            
class mnistDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self,samples_per_digit, normalize, file):
        self.name = 'MNIST'
        self.file = file
        self.normalize = normalize
        self.num_classes = 10 # Digit 0-9
        
        mnist = pd.read_csv("datasets/MNIST/mnist_test.csv").values  
        vectorsF = []
        labelsF = []
        for i in range(0,10):
            mask = mnist[:,-1]==i
            temp = mnist[mask]
            #temp = temp/255
            np.random.shuffle(temp)
            vectorsT = temp[0:samples_per_digit, :-1] / 255
            labelsT = temp[0:samples_per_digit, -1]
            if i == 0:
                vectorsF = vectorsT
            else:
                vectorsF = np.concatenate((vectorsF, vectorsT), axis=0)
            labelsF = np.append(labelsF, labelsT, axis=0)
        
        self.labels = labelsF
        self.vectors = vectorsF
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data(self.vectors)
         
