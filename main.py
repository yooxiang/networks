'''
vertex - list of vertex 
vertex_attach - list of list of vertex to attached 
vertex_attach_count - len of the lists
'''
import random
import matplotlib.pyplot as plt
import scipy as sp 
import networkx as nx
import numpy as np
from datetime import datetime
from tqdm import tqdm
startTime = datetime.now()

def logbin(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y

def Pure_attachment(vertex_attach_count):
    return random.choice(vertex_attach_count)

def generate_edge(vertex_attach):
    edge = []
    for i in range(len(vertex_attach)):
        for j in range(len(vertex_attach[i])):
            edge.append([i,vertex_attach[i][j]])
    return edge
        
    
def Network(N,model):
    vertex = ([0,1])
    vertex_attach = ([[1],[0]])
    vertex_attach_count = ([1,0])
    
    for i in tqdm(range (N - 2)):
        vertex.append(i+2)

        new_vertex = model(vertex_attach_count)
        vertex_attach.append([new_vertex])
        vertex_attach[vertex.index(new_vertex)].append(i+2)
        vertex_attach_count.append(new_vertex)
        vertex_attach_count.append(i+2)
        
    return vertex, vertex_attach , vertex_attach_count

def Network_visual(N,model):
    vertex, vertex_attach , vertex_attach_count = Network(N,model)
    G = nx.Graph()
    G.add_edges_from(generate_edge(vertex_attach))
    G.add_nodes_from(vertex)
    nx.draw(G, node_color = 'k', node_size = 1, node_shape = 'o', linewidth = 0.01)
    plt.title("Network with")
    plt.savefig("exaple", dpi = 1000)
    plt.show()

def Network_order_distrabution(N,model):
    vertex, vertex_attach , vertex_attach_count = Network(N,model)
    n_k = []
    for i in range(len(vertex_attach)):
        n_k.append(len(vertex_attach[i]))
    x,y = logbin(n_k, scale = 1.2, zeros = False)
    plt.loglog(x,y, color = 'k',marker = 'x', linewidth = 0)
    plt.xlabel("Order")
    plt.ylabel("Probability")
    plt.title("Logbin of probability distrabution")
    g = sp.poly1d(sp.polyfit(sp.log10(x),sp.log10(y),1))
    plt.plot(x, 10**g(sp.log10(x)), color = 'k', linewidth = 0.2)
    print(datetime.now() - startTime, "for", N, "nodes")
    
Network_order_distrabution(100000,Pure_attachment)
