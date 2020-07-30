# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 30 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import gtls
import numpy as np
from episodic_gwr import EpisodicGWR
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

def replay_samples(net, size) -> (np.ndarray, np.ndarray):
    samples = np.zeros(size)
    r_weights = np.zeros((net.num_nodes, size, net.dimension))
    r_labels = np.zeros((net.num_nodes, len(net.num_labels), size))
    for i in range(0, net.num_nodes):
        for r in range(0, size):
            if r == 0: samples[r] = i
            else: samples[r] = np.argmax(net.temporal[int(samples[r-1]), :])
            r_weights[i, r] = net.weights[int(samples[r])][0]
            for l in range(0, len(net.num_labels)):
                r_labels[i, l, r] = np.argmax(net.alabels[l][int(samples[r])])
    return r_weights, r_labels
        
if __name__ == "__main__":

    train_flag = True
    train_type = 1 # 0:Batch, 1: Incremental
    train_replay = True
    
    ds_iris = gtls.IrisDataset(file='datasets/iris.csv', normalize=True, numClass=3)
    ds_iris2 = gtls.IrisDataset(file='datasets/iris.csv', normalize=True, numClass=3)
    print("%s from %s loaded." % (ds_iris.name, ds_iris.file))
    print("%s from %s loaded." % (ds_iris2.name, ds_iris2.file))

    assert train_type < 2, "Invalid type of training."
    
    '''
    MC-DNM supports multi-class neurons.
    Set the number of label classes per neuron and possible labels per class
    e.g. e_labels = [50, 10]
    is two labels per neuron, one with 50 and the other with 10 classes.
    Setting the n. of classes is done for experimental control but it is not
    necessary for associative GWR learning.
    '''
    e_labels = [10, 10]
    s_labels = [10]
    ds_vectors = ds_iris.vectors
    ds_labels = np.zeros((len(e_labels), len(ds_iris.labels)))
    ds_labels[0] = ds_iris.labels
    ds_labels[1] = ds_iris.labels
    
    e_labels2 = [10, 10]
    s_labels2 = [10]
    ds_vectors2 = ds_iris2.vectors
    ds_labels2 = np.zeros((len(e_labels2), len(ds_iris2.labels)))
    ds_labels2[0] = ds_iris2.labels
    ds_labels2[1] = ds_iris2.labels
    
    num_context = 2 # number of context descriptors
    epochs = 1 # epochs per sample for incremental learning
    a_threshold = [0.85, 0.80] # iris 0.85,0.80
    beta = 0.7
    learning_rates = [0.2, 0.001] # [0.2, 0.001] 
    context = True

    g_episodic = EpisodicGWR()
    g_episodic.init_network_mc(ds_iris, ds_iris2, e_labels, num_context)
    
    g_semantic = EpisodicGWR()
    g_semantic.init_network_mc(ds_iris, ds_iris2, s_labels, num_context)
    
    ds_vectors3 = np.hstack((ds_vectors, ds_vectors2))
    
    if train_type == 0:
        # Batch training
        # Train episodic memory
        g_episodic.train_egwr(ds_vectors, ds_labels, epochs, a_threshold[0],
                              beta, learning_rates, context, regulated=0)
                              
        e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels,
                                              ret_vecs=True)
        # Train semantic memory
        g_semantic.train_egwr(e_weights, e_labels, epochs, a_threshold[1], beta, 
                          learning_rates, context, regulated=1)        
    else:
        # Incremental training
        n_episodes = 0
        batch_size = 10 # number of samples per epoch:10
        # Replay parameters
        replay_size = (num_context * 2) + 1 # size of RNATs
        replay_weights = []
        replay_labels = []
        
        # Train episodic memory
        for s in range(0, ds_vectors.shape[0]):
            print("Iteration %s" % s)
          
            g_episodic.train_egwr_mc(ds_vectors3[s:s+batch_size],
                                  ds_labels[:, s:s+batch_size], ds_vectors.shape[1],
                                  epochs, a_threshold[0], beta, learning_rates,
                                  context, regulated=0)
            
            e_weights, e_labels = g_episodic.test_mc(ds_vectors3, ds_labels, ds_vectors.shape[1],
                                                  ret_vecs=True)
            # Train semantic memory
            g_semantic.train_egwr(e_weights[s:s+batch_size],
                                  e_labels[:, s:s+batch_size],
                                  epochs, a_threshold[1], beta, learning_rates,
                                  context, regulated=1)
                                  
            if train_replay and n_episodes > 0:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    g_episodic.train_egwr_mc(replay_weights[r], replay_labels[r, :], ds_vectors.shape[1],
                                          epochs, a_threshold[0], beta,
                                          learning_rates, 0, 0)
                    
                    g_semantic.train_egwr(replay_weights[r], replay_labels[r],
                                          epochs, a_threshold[1], beta, 
                                          learning_rates, 0, 1)
            
            # Generate pseudo-samples
            if train_replay: 
                replay_weights, replay_labels = replay_samples(g_episodic, 
                                                               replay_size)
            n_episodes += 1
           
        
    g_episodic.test_mc(ds_vectors3, ds_labels, ds_vectors.shape[1], test_accuracy=True)
    g_semantic.test_mc(e_weights, e_labels, ds_vectors.shape[1], test_accuracy=True)
    
    print("Accuracy episodic: %s, semantic: %s" % 
          (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))
    
    gtls.plot_network(g_episodic, edges=True, labels=True)
    gtls.plot_network(g_semantic, edges=True, labels=True)
    #gtls.export_network("irisModel.n5", g_semantic)
    #g_semantic2 = gtls.import_network("irisModel.n5", EpisodicGWR)
    #g_semantic2.train_egwr
    
    