import numpy as np

#
# This script contains functions to calculate the 1D lagrange polynomial in a certain range
#

def EvalBasis(p, neval, xeval,xmin,xmax):
    '''
    Function: EvalBasis
    -------------------
    This function evaluates the Lagrange basis polynomials at 
    specified coordinates (xeval) in the reference domain [xmin,xmax].
    The Lagrange nodes are assumed to be equidistant.

    INPUTS:
        p: polynomial order
        neval: number of points at which to evaluate
        xeval: coordinates (in reference space) at which to evaluate

    OUTPUTS:
        phi: evaluated basis data - size [neval,p+1]
    '''

    nnode = p + 1 # number of Lagrange nodes
    phi = np.empty([neval,nnode])
#    phi_d = np.empty([neval,nnode])
    # p = 0 case is trivial
    if p == 0:
        phi[:,0] = 1.
#        phi_d[:,0]=0.
        return phi

    # Equidistant Lagrange nodes
    xnode = np.zeros(nnode)
    dx = (xmax-xmin)/float(p)
    for i in range(nnode): 
        xnode[i] = float(i)*dx # Lagrange node positions
    for i in range(neval):
        phi[i,:] = BasisLagrange1D(xeval[i], xnode, nnode, phi[i,:])
#        phi_d[i,:] = BasisLagrange1DDerivative(xeval[i],xnode,nnode,phi[i,:])
    return phi


def BasisLagrange1D(x, xnode, nnode, phi):
    '''
    Function: BasisLagrange1D
    -------------------
    This function evaluates the Lagrange basis polynomials at a
    specified coordinate (x) in the domin [xmin,xmax].
    The Lagrange nodes are stored in xnode.

    INPUTS:
        x: coordinate (in reference space) at which to evaluate
        xnode: coordinates (in reference space) of Lagrange nodes
        nnode: number of Lagrange nodes

    OUTPUTS:
        phi: value(s) of Lagrange basis at x
    '''

    for j in range(nnode):
    	pj = 1.
    	for i in range(j): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
    	for i in range(j+1,nnode): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
    	phi[j] = pj

    return phi


