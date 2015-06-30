
import os
import numpy as np

from seisflows.tools import unix

from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import loadtxt, savetxt



class NLCG:
    """ Nonlinear conjugate gradient method
    """

    def __init__(self, path, thresh, itermax):
        self.path = path
        self.itermax = itermax
        self.thresh = thresh

        try:
            self.iter = loadtxt(self.path+'/'+'NLCG/iter')
        except IOError:
            unix.mkdir(self.path+'/'+'NLCG')
            self.iter = 0


    def __call__(self):
        """ Returns NLCG search direction
        """
        self.iter += 1
        savetxt(self.path+'/'+'NLCG/iter', self.iter)

        unix.cd(self.path)
        g_new = loadnpy('g_new')

        if self.iter == 1:
            return -g_new, 0

        elif self.iter > self.itermax:
            print 'restarting NLCG... [periodic restart]'
            self.restart()
            return -g_new, 1

        # compute search direction
        g_old = loadnpy('g_old')
        p_old = loadnpy('p_old')
        beta = pollak_ribere(g_new, g_old)
        p_new = -g_new + beta*p_old

        # check restart conditions
        if check_conjugacy(g_new, g_old) > self.thresh:
            print 'restarting NLCG... [loss of conjugacy]'
            self.restart()
            return -g_new, 1

        elif check_descent(p_new, g_new) > 0.:
            print 'restarting NLCG... [not a descent direction]'
            self.restart()
            return -g_new, 1

        else:
            return p_new, 0


    def restart(self):
        """ Restarts algorithm
        """
        self.iter = 1
        savetxt(self.path+'/'+'NLCG/iter', self.iter)



### utility functions

def fletcher_reeves(g_new, g_old):
    num = np.dot(g_new, g_new)
    den = np.dot(g_old, g_old)
    beta = num/den
    return beta

def pollak_ribere(g_new, g_old):
    num = np.dot(g_new, g_new-g_old)
    den = np.dot(g_old, g_old)
    beta = num/den
    return beta

def check_conjugacy(g_new, g_old):
    return abs(np.dot(g_new, g_old) / np.dot(g_new, g_new))

def check_descent(p_new, g_new):
    return np.dot(p_new, g_new) / np.dot(g_new, g_new)




