import cv2
import cmath
from math import pi, log, ceil
import numpy as np
from fractions import Fraction
import getopt
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import timeit

"""
collum-wise DFT: on a vector, behaves like a 1D DFT, on a matrix applies a DFT to each collum
"""
def dft_col(x):
    l = x.shape[0]
    coeffs = np.array([[cmath.exp(-2*pi*n*k*1j/l) for n in range(0, l)] for k in range(0, l)])
    return np.matmul(coeffs, x)

"""
row-wise DFT: on a row matrix, behaves like a 1D DFT, on a matrix applies a DFT to each row
"""
def dft_row(x):
    l = x.shape[1]
    coeffs = np.array([[cmath.exp(-2*pi*n*k*1j/l) for n in range(0, l)] for k in range(0, l)])
    return np.matmul(x, coeffs)

"""
DFT on a python list, numpy array, or numpy matrix of any size. will return a numpy array if a list is passed as argument
"""
def dft(x):
    if isinstance(x, list):
        x = np.array(x)
    if x.shape[0] > 1:
        x = dft_col(x)
    if len(x.shape)>1 and x.shape[1] > 1:
        x = dft_row(x)
    return x

"""
cooley tukey implementation of collum-wise dft
"""
def ctdft_col(x):
    l = x.shape[0]
    if l % 2 != 0:
        return dft_col(x)
    evens = ctdft_col(x[0:l:2])
    odds = ctdft_col(x[1:l:2])
    result = np.empty((l, x.shape[1]), dtype = 'complex_')
    for k in range(0, l//2): # because cis(theta) is periodic, k+N/2 is cis(theta+pi), so even component is the same and odd component is negated
        odd = odds[k] * cmath.exp(-2*pi*k*1j/l) 
        result[k] = evens[k] + odd
        result[k+l//2] = evens[k] - odd
    return result

"""
cooley tukey implementation of row-wise dft
"""
def ctdft_row(x):
    l = x.shape[1]
    if l % 2 != 0:
        return dft_row(x)
    evens = ctdft_row(x[:,0:l:2])
    odds = ctdft_row(x[:,1:l:2])
    result = np.empty((x.shape[0], l), dtype = 'complex_')
    for k in range(0, l//2):
        odd = odds[:,k] * cmath.exp(-2*pi*k*1j/l)
        result[:,k] = evens[:,k] + odd
        result[:,k+l//2] = evens[:,k] - odd
    return result

"""
cooley tukey implementation mirroring the naive implementation, but calling cooley tukey fxns
"""
def ctdft(x):
    if isinstance(x, list):
        x = np.array(x)
    if x.shape[0] > 1:
        x = ctdft_col(x)
    if len(x.shape)>1 and x.shape[1] > 1:
        x = ctdft_row(x)
    return x

"""
inverts a fourier transform
rather than reimplementing the entire function for 1D/2D cases,
this negates the imaginary coordinates on the input and output of DFT,
which has the same result as preforming the IDFT with its negated powers,
then divides each result by N
"""
def ictdft(x):
    if isinstance(x, list):
        x = np.array(x)
    f = lambda c: c.real - c.imag*1j
    result = ctdft(f(x))
    return f(result)/x.size

"""
displays an image on the given plot as normalized values on a logarithmic scale.
if a size is given, can resize the image as well
"""
def show_normalized(plt, img, size = None):
    img = abs(img)
    if size != None:
        img = cv2.resize(img, size)
    plot = plt.imshow(img, norm=LogNorm(img.min(), img.max()), cmap="gray")
    fig.colorbar(plot, ax=plt)

"""
denoises the image by zero-ing the highest frequencies in a cross pattern
eg.
.00.
0000
0000
.00.
would be the result on a 4x4 image with amt = 0.5;
amt dictates the fraction of image width & height zero'd by the cross pattern
the percent of remaining non-zeroes will be aprox. amt^2 of coefficients
"""
def denoise(img, amt):
    c,r = img.shape
    dc = round(c/2*amt)
    dr = round(r/2*amt)
    num_zeros = 2*dc*r + 2*dr*c - 4*dr*dc
    print(f"Using {img.size-num_zeros} non-zeroes; {(img.size-num_zeros)*100/img.size}% of original fourier coefficients")
    img[c//2-dc:c//2+dc,:] = np.zeros((dc*2, r), dtype = 'complex_')
    img[:,r//2-dr:r//2+dr] = np.zeros((c, dr*2), dtype = 'complex_')
    return img

if __name__ == "__main__":
    (opts, args) = getopt.getopt(sys.argv[1:], "i:m:")
    imgfile = "moonlanding.png"
    mode = 1
    for (o, v) in opts:
        if o == "-m":
            assert v.isdigit(), f"Mode must be an integer"
            v = int(v)
            assert v in range(1, 5), f"Mode should be 1, 2, 3, or 4"
            mode = v
        if o == "-i":
            imgfile = v
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)/256
    orig_size = img.shape[1], img.shape[0] # the image dimensions from what matplotlib & opencv used are in the oppossite order as what numpy treats the matrix dimensions
    size =  2**ceil(log(img.shape[1], 2)), 2**ceil(log(img.shape[0], 2)) # power of 2 image dimensions

    img_scaled = cv2.resize(img, size) # scale to power of 2s to allow FDFT
    
    if mode == 1:
        fig, (plt_orig, plt_dft) = plt.subplots(1, 2)
        fig.suptitle("Log scaled |DFT|")
        img_dft = ctdft(img_scaled)
        plt_orig.imshow(img_scaled, cmap="gray") # this is using the power 2 scaled image to display because that is what is being used to fetch the DFT values
        show_normalized(plt_dft, img_dft) # again, don't scale image back to original ratio, show most accurate DFT display
    elif mode == 2:
        denoise_amount = 0.8 # amount to denoise image by (see denoise function)
        
        fig, (plt_orig, plt_denoise) = plt.subplots(1, 2)
        fig.suptitle("Denoising")
        img_dft = denoise(ctdft(img_scaled), denoise_amount)
        plt_orig.imshow(img, cmap="gray")
        img_denoise = cv2.resize(abs(ictdft(img_dft)), orig_size) # resize back image to preserve ratio with original
        plt_denoise.imshow(img_denoise, cmap="gray")
    elif mode == 3:
        compress_amounts = ((0, 0.5, 0.75), (0.9, 0.99, 0.999)) # compression ratios for each image, in reading order. note actual compression amounts may vary somewhat with inexact numbers of pixels to represent
        
        fig, plts = plt.subplots(2, 3)
        fig.suptitle("Compression")
        img_dft = ctdft(img_scaled)
        threshs = abs(img_dft).flatten()
        threshs.sort() # sort the values by magnitude to allow retrieving the nth percentile threshhold
        for i in (0,1):
            for j in (0,1,2):
                if i==0 and j==0: # the first compression amount will always be ignored, just print original image
                    print(f"Original image: {threshs.size} non-zeroes")
                    plts[i][j].imshow(img, cmap="gray")
                    continue
                ti = round(compress_amounts[i][j]*threshs.size) # the threshhold index comes from the compression amount as a fraction of all the possible sorted values; any higher values will be kept
                thresh = threshs[ti]
                print(f"Magnitude threshhold with {threshs.size - ti} non-zeroes; {100*(threshs.size - ti)/threshs.size}% of fourier coefficients remain")
                f = np.vectorize(lambda c: c if abs(c)>=thresh else 0) # zero values who's magnitude is below threshhold
                img_dft = f(img_dft)
                img_comp = cv2.resize(abs(ictdft(img_dft)), orig_size) # resize back image to preserve ratio with original
                plts[i][j].imshow(img_comp, cmap="gray")
        
    elif mode == 4:
        rng = np.random.default_rng()
        sizes = [2**p for p in range(5, 11)] # matrix sizes to test algorithms on. should be powers of 2 to accurately portray CTDFT runtime
        dft_avg = []
        dft_dev = []
        ctdft_avg = []
        ctdft_dev = []
        for s in sizes:
            dft_runs = []
            ctdft_runs = []
            print(f"Matrix size {s}x{s}") # progress output so program doesn't appear to hang
            for t in range(0,10):
                print(f"\tRun {t+1}") # progress output so program doesn't appear to hang
                a = rng.random((s, s))
                start = timeit.default_timer()
                dft(a)
                stop = timeit.default_timer()
                dft_runs.append(stop - start)

                start = timeit.default_timer()
                ctdft(a)
                stop = timeit.default_timer()
                ctdft_runs.append(stop - start)
            dft_avg.append(sum(dft_runs)/10)
            dft_dev.append(2*np.std(dft_runs))
            ctdft_avg.append(sum(ctdft_runs)/10)
            ctdft_dev.append(2*np.std(ctdft_runs))
        plt.errorbar(sizes, dft_avg, dft_dev, ecolor='k', label="Naive DFT")
        plt.errorbar(sizes, ctdft_avg, ctdft_dev, ecolor='k', label="Cooley Tukey DFT")
        plt.legend()
        plt.xlabel("matrix dimension (NxN)")
        plt.ylabel("time (s)")
    plt.show()
