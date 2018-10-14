
# coding: utf-8

# In[2]:


import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 10, 6
subSize=100


# In[17]:


def getDegrees(G):
    # sum of in and out degrees by default. use in_degree for specific
    return list(nx.degree(G))

def compareDirect(a,b,size=10,write=""):        
    print("Calculated :\t",a[:size],"...")
    print("")
    print("Expected :\t",b[:size],"...")    
    print("")
#     diffs=[ abs(a-b) for a,b in zip(a,b)]
#     print("Absolute Diffs: ",diffs[:size],"..." if size<len(a) else "")
#     print("")
#     print("Sum of absolute diffs: ",sum(diffs))
#     ratios=[ a/b if b else '0' for a,b in zip(a,b)]
#     print("Ratios: ",ratios)
    if(write!=""):
        a=sorted(a,reverse=True)
        b=sorted(b,reverse=True)
        pd.DataFrame(a).to_csv(write+"_code.csv",header=False,index=None)
        pd.DataFrame(b).to_csv(write+"_tool.csv",header=False,index=None)
        
    print("\n Spearman rank correlation: ")
    print(stats.spearmanr(a,b))
    
    
    
def compare(a,b,name,size=10):
    a=sorted(a,key=lambda x:x[1],reverse=True)
    b=sorted(b,key=lambda x:x[1],reverse=True)
    
    pd.DataFrame(a).to_csv(name+"_code.csv",header=False,index=None)
    pd.DataFrame(b).to_csv(name+"_tool.csv",header=False,index=None)
    
    a=[x[1] for x in a]
    b=[x[1] for x in b]
    compareDirect(a,b,size)


# In[18]:


diG=nx.read_weighted_edgelist("higgs-mention_network.edgelist",create_using=nx.DiGraph())#read_edgelist("com-dblp.ungraph.txt")
N=diG.number_of_nodes()
print("Originally: "+str(N)+" nodes")

degrees=sorted(getDegrees(diG),key=lambda x:x[1])
subG=diG.subgraph([x[0] for x in degrees[N-subSize:]])
# subG=diG.subgraph(np.random.choice(list(subG.nodes),size=subSize, replace=False))
subGdegrees=getDegrees(subG)
subN=subG.number_of_nodes()
print("Taken: "+str(subN)+" random nodes from "+str(subSize)+" highest degree nodes")


# In[19]:


G=subG
if(subN<200):    
    pos=dict.fromkeys(G)
    for d in pos:
        pos[d]=(int(20*np.random.rand()),int(20*np.random.rand())) if d!='3998' else (0,0)
    nx.draw_networkx(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()


# In[20]:


"""
Degree Centrality = degree(n) = In undirected graph it is no of edges 
"""
subNminus=subN-1
degreeCentrality=[(x[0],x[1]/subNminus) for x in subGdegrees]
compare(degreeCentrality,list(nx.degree_centrality(subG).items()),name="degree_centrality")


# In[21]:


# Get all pairs shortest path lengths
all_spls=nx.floyd_warshall(subG) #floyd_warshall gives inf too


# In[22]:


"""
Closeness centrality -  is the average length of the shortest path between the node and all other nodes in the graph i.e.
in the connected part of graph containing the node. If the graph is not completely connected, 
this algorithm computes the closeness centrality for each connected part separately.
>> Normalized again with a factor n-1/N-1
"""

closeness=[]
for n,lengths in all_spls.items():
    component=[x for x in lengths.values()  if x!=np.inf ]
    sumDists=sum(component)
    lminus=len(component)-1
    n_fac=lminus/subNminus
    closeness.append( (n,(lminus / sumDists * n_fac) if(sumDists!=0) else 0.0) )
    
#From the tool
ideal=nx.closeness_centrality(subG,distance='weight') #consider the weights

compare(closeness,list(ideal.items()),name="closeness")


# In[12]:


"""
Betweenness centrality computed as follows:
1. For each pair of vertices (s,t), compute the shortest paths between them. <-- spaths
2. For each pair of vertices (s,t), determine the fraction which pass through the vertex v. <-- iterating on paths
3. Sum this fraction over all pairs of vertices (s,t). <-- sum while iterating

Betweenness(v) = no of SPs passing thru v/Total no of SPs
"""
from itertools import count
from heapq import heappush, heappop

def accumulate_from_S(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff # count paths
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

def single_source_dijkstra_multi_sps(G, s, weight='weight'):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            
            # """ Multiple paths handled here"""
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
                
    return S, P, sigma
                
betweenness = dict.fromkeys(subG, 0.0)  
for s in subG:
    S, P, sigma = single_source_dijkstra_multi_sps(subG, s, 'weight')
    betweenness = accumulate_from_S(betweenness,S,P,sigma,s)

compare(list(betweenness.items()),list(nx.betweenness_centrality(subG,normalized=False,weight='weight').items()),name="betweenness")


# In[13]:


# G=nx.path_graph(4)
# eigenvector_centrailty=power_iteration(G,num_simulations=30)

# compareDirect(eigenvector_centrailty,list(nx.eigenvector_centrality(G,weight='weight').values()))#,size=subSize)


# In[24]:


"""
Eignevector Centrality
Concept: connections to high-scoring nodes contribute more to the score of the node in question
         than equal connections to low-scoring nodes.
The Eigenvector corresponding to largest eigenvalue gives the desired centrality
For this we use the power method
It is an eigenvalue algorithm: given a diagonalizable matrix A, the algorithm will give greatest(in absolute value) eigenvalue of A,
and a nonzero vector v, the corresponding eigenvector. The algorithm is also known as the Von Mises iteration.[1]

Power iteration is a very simple algorithm, but it may converge slowly. It is suitable for large sparse matrices
"""


def power_iteration(G, num_simulations):
    A=nx.to_numpy_matrix(G);
    A=np.array(A)
    EPSILON = np.longdouble(1e-9)
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
#     while 1:
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)        
        b_k_next = b_k1 / b_k1_norm
        if(np.sum(abs(b_k_next-b_k))<EPSILON*len(b_k)):
            break
        b_k=b_k_next
        num_simulations-=1;
    print("left iters: ",num_simulations)
    return b_k_next

eigenvector_centrality=power_iteration(subG,num_simulations=30)

compare(list(zip(G.nodes,eigenvector_centrality)),list(nx.eigenvector_centrality(subG,weight='weight').items()),name="eigenvector")#,size=subSize)


# In[26]:


"""
Clustering Coefficient 
The global clustering coefficient is the number of closed triplets (or 3 x triangles) 
over the total number of triplets (both open and closed). 

The local clustering coefficient is the proportion of links between the vertices 
within its neighbourhood divided by the number of links that could possibly exist between them.

Average clustering coefficient is mean of local clusterings
"""
unsubG=nx.to_undirected(subG)
clustering_coeffs={}
for n in unsubG:
    nghs = [e[1] for e in unsubG.edges(n)]
    d=len(nghs)
    e=subG.subgraph(nghs).number_of_edges()    
    # for directed..
    clustering_coeffs[n]= 1*e/(d*(d-1)) if d>1 else 0.0;

compare(list(clustering_coeffs.items()),list(nx.clustering(unsubG).items()),name="clustering")#,size=subSize)


# In[16]:


compareDirect(list(clustering_coeffs.values()),list(nx.clustering(unsubG).values()),write="clustering")


# In[63]:



# N=3
# G=nx.grid_2d_graph(N,N)
# pos = dict( (n, n) for n in G.nodes() )
# labels = dict( ((i, j), i + (N-1-j) * N ) for i, j in G.nodes() )
# nx.relabel_nodes(G,labels,False)
# inds=labels.keys()
# vals=labels.values()
# inds=sorted(inds)
# vals=sorted(vals)
# pos2=dict(zip(vals,inds))
# nx.draw_networkx(G, pos=pos2, with_labels=False, node_size = 30)
# plt.show()

