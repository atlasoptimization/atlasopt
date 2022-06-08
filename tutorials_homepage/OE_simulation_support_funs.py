def Simulation_random_field(cov_x, cov_y, grid_x, grid_y, explained_var):
          
    """
    The goal of this function is to simulate a realization of a random field
    efficiently employing the tensor product nature of covariance functions.
    This does not work for all random field but only for those, whose covariance
    function cov((x_1,y_1),(x_2,y_2)) decomposes as cov_x(x_1,x_2)*cov_y(y_1,y_2). 
    The actual simulation uses the Karhunen Loewe expansion of a process into
    superpositions of basis functions weighted by the eigenvalues of the covariance
    matrix multiplied with white noise variables.
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Simulate and assemble solution
        
    INPUTS
    The inputs consist in the two covariance functions whose product forms the
    multivariate covariance of the random field. Furthermore grid values for
    the input coordinates are provided and a number between 0 and 1 indicating
    how many terms are used in the superposition of the Karhunen Loewe expansion.
    
    Name                 Interpretation                             Type
    cov_x               Function handle for the cov function       function handle
                        Maps two real numbers x_1,x_2 to a real
                        number indicating the cov in x direction
    cov_y               Function handle for the cov function       function handle
                        Maps two real numbers y_1,y_2 to a real
                        number indicating the cov in y direction
    grid_x              Matrix containing info w.r.t the x vals    Matrix [n,n]
                        at each location for which a value is
                        to be simulated
    grid_y              Matrix containing info w.r.t the y vals    Matrix [n,n]
                        at each location for which a value is
                        to be simulated
    explained_var       The fraction of variance to be explained   Number in [0,1]
                        by the simulation. The closer to 1, the 
                        more faithful the reproduction of the cov
                        structure and the longer the runtime
                        
    OUTPUTS
    The outputs consist in the matrix Random_field which is a realization of the
    random field from which a sample was supposed to be drawn.
    
    Name                 Interpretation                             Type
    Random_field        Realization of the random field            Matrix [n,n]

    Written by Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
    
    """
    
    
    
    """
        1. Definitions and imports -------------------------------------------
    """
    
    
    # i) Import packages
    
    import numpy as np
    import numpy.linalg as lina
    
    
    # ii)) Define auxiliary quantities
    
    n_y,n_x=np.shape(grid_x)
    
    
    
    
    """
        2. Set up problem matrices -------------------------------------------
    """
    
    
    # i) Component covariance matrices
    
    K_x=np.zeros([n_x,n_x])
    K_y=np.zeros([n_y,n_y])
    for k in range(n_x):
        for l in range(n_x):
            K_x[k,l]=cov_x(grid_x[0,k], grid_x[0,l])
            
    for k in range(n_y):
        for l in range(n_y):
            K_y[k,l]=cov_y(grid_y[k,0], grid_y[l,0])
                
    [U_x,S_x,V_x]=lina.svd(K_x)
    [U_y,S_y,V_y]=lina.svd(K_y)
    

    # ii) Indexing and ordering of eigenvalues
    
    lambda_mat=np.outer(S_y, S_x)
    index_mat_ordered=np.unravel_index(np.argsort(-lambda_mat.ravel()), [n_y,n_x])
    lambda_ordered=lambda_mat[index_mat_ordered]
    
    lambda_tot=np.sum(lambda_mat)
    lambda_cumsum=np.cumsum(lambda_ordered)
    stop_index=(np.where(lambda_cumsum>=explained_var*lambda_tot))[0][0]
    
        
    
    """
        3. Simulate and assemble solution ------------------------------------
    """
        
    
    # i) Iterative Karhunen Loewe composition
    
    white_noise=np.random.normal(0,1,[stop_index])
    
    Random_field=np.zeros([n_y,n_x])
    for k in range(stop_index):
        Random_field=Random_field+white_noise[k]*lambda_ordered[k]*np.outer(U_y[:,index_mat_ordered[0][k]],U_x[:,index_mat_ordered[1][k]])
              
        
        
    # ii) Return solution
    
    return Random_field, K_x, K_y
