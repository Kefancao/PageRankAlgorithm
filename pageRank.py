import numpy as np
from scipy.sparse import dok_matrix
from copy import deepcopy
import matplotlib.pyplot as plt


# Sparse Matrix Multiplication is used to avoid unnecessary matrix multiplications (mult by 0) that
#  could take up a lot of time. 
def SparseMatMult(G, x):
    '''
      y = SparseMatMult(G, x)
      
      Multiplies a vector (x) by a sparse matrix G,
      such that y = G @ x .
      
      Inputs:
        G is an NxM dictionary-of-keys (dok) sparse matrix
        x is an M-vector
      
      Output:
        y is an N-vector
    '''
    # Exception handling for when columns of G are not the same length as rows of x
    if np.shape(G)[1] != np.shape(x)[0]:
      raise Exception("The number of columns of G must equal the number of rows of x")
    
    # Performs a matrix-vector multiplication, where G is a sparse matrix
    y = np.zeros(np.shape(G)[0], dtype=float)
    
    # Generate the rows and columns as a tuple of nonzero indices
    non_zeroes = G.nonzero()
    # Used to detect when the next nonzero index is a new row
    prev = non_zeroes[0][0]
    sum = 0
    for i in range(len(non_zeroes[0])):
        # If the next nonzero index is a new row, add the sum to the previous row
        if i != prev:
            y[prev] = sum
            sum = 0
        sum += G[non_zeroes[0][i], non_zeroes[1][i]] * x[non_zeroes[1][i]][0]
        prev = non_zeroes[0][i]
    # Add the last sum to the last nonzero indexed row
    y[prev] = sum
    
    return np.array([y]).T
  
  
  def PageRank(G, alpha, maxIter=10000):
    '''
     p, iters = PageRank(G, alpha)

     Computes the Google Page-rank for the network in the adjacency matrix G.
     
     Note: This function never forms a full RxR matrix, where R is the number
           of node in the network.

     Input
       G     is an RxR adjacency matrix, G[i,j] = 1 iff node j projects to node i
             Note: G must be a dictionary-of-keys (dok) sparse matrix
       alpha is a scalar between 0 and 1

     Output
       p     is a probability vector containing the Page-rank of each node
       iters is the number of iterations used to achieve a change tolerance
             of 1e-8 (changes to elements of p are all smaller than 1e-8)
    '''
    G = deepcopy(G)
    # Constructs the initial matrix of Page-rank probabilities
    d = np.ones(np.shape(G)[0], dtype=float)
    # Get the non-zero elements of the sparse matrix and zip them
    non_zeroes = G.nonzero()
    non_zeroes = list(zip(non_zeroes[0], non_zeroes[1]))
    for entry in non_zeroes:
        G[entry[0], entry[1]] = 1/G.getcol(entry[1]).count_nonzero()
        d[entry[1]] = 0 

    # Constructing the e and d matrices
    R = np.shape(G)[0]
    e = dok_matrix((R, 1), dtype=np.float32)
    for i in range(R):
        e[i, 0] = 1
    
    d_transpose = dok_matrix((1, R), dtype=np.float32)
    for i in range(R):
        d_transpose[0, i] = d[i]
    
    # Computes the M matrix
    Q = G + (1/R) * (e @ d_transpose)
    
    # Uniform p as per the requirements
    p = [1/R for i in range(R)]
    
    # The second half of the matrix M
    prob_iter = [(1-alpha)/R for i in range(R)]
    
    # Multiplies until the infinity norm is less than 10^-8
    for iters in range(maxIter):
        p_prev = deepcopy(p)
        p = alpha * Q @ p + prob_iter
        if np.linalg.norm(p - p_prev, ord=np.inf) <= 1e-8:
            return p, iters + 1
    return 'The maximum number of iterations was reached'
 

# For testing purposes of the graph in the README file
G = dok_matrix((11,11), dtype=np.float32)

# The clique on the left with {0, 1, 2, 3, 4}
G[0,1] = 1
G[1,0] = 1
G[0,2] = 1
G[2,0] = 1
G[0,3] = 1
G[3,0] = 1
G[4,0] = 1
G[0,4] = 1
G[1,2] = 1
G[2,1] = 1
G[1,3] = 1
G[3,1] = 1
G[1,4] = 1
G[4,1] = 1
G[2,4] = 1
G[2,3] = 1
G[3,2] = 1
G[4,3] = 1
G[4,2] = 1
G[3,4] = 1

# Subgraph on the right with {5, 6, 7, 8, 9, 10}
G[6,5] = 1
G[5,6] = 1
G[7,5] = 1
G[5,7] = 1
G[8,5] = 1
G[5,8] = 1
G[9,5] = 1
G[5,9] = 1
G[10,5] = 1
G[5,10] = 1

# Link two subnets together
G[1,5] = 1

G[10,1] = 1
G[1,10] = 1


# Plotting the node scores.
p_vec = PageRank(G, 0.85)[0]
# Plot the stem plot with nodes from 0 to 10 and the scores from p_vec and label the axes
plt.stem(np.arange(11), p_vec, use_line_collection=True)
plt.title('Stem Plot of Page Rank Scores with alpha = 0.85')
plt.xlabel("Nodes")
plt.ylabel("Scores")



  
