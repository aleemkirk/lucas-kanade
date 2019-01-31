#implemntation of lucas kanade using code from https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/

import numpy as np
from scipy import signal
from numpy.linalg import inv
import cv2 as cv2

def optical_flow(I1g, I2g, window_size=7, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g #/ 255. # normalize pixels
    I2g = I2g #/ 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -1*kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            A = np.concatenate((Ix.transpose(), Iy.transpose()), axis=1)
            b = -1*It.transpose()
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            c = inv(np.dot(A.transpose(), A))
            d = np.dot(A.transpose(), b)
            nu = np.dot(c, d) # get velocity here
            u[i,j]=nu[0]
            v[i,j]=nu[1]
 
    return (u,v)

if(__name__ == "__main__"):
    basketball1 = cv2.cvtColor(cv2.imread('basketball1.png'), cv2.COLOR_RGB2GRAY)
    basketball2 = cv2.cvtColor(cv2.imread('basketball2.png'), cv2.COLOR_RGB2GRAY)
    print(optical_flow(basketball1, basketball2)[0])
    