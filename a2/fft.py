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

def dft_col(x):
    l = x.shape[0]
    coeffs = np.array([[cmath.exp(-2*pi*n*k*1j/l) for n in range(0, l)] for k in range(0, l)])
    return np.matmul(coeffs, x)

def dft_row(x):
    l = x.shape[1]
    coeffs = np.array([[cmath.exp(-2*pi*n*k*1j/l) for n in range(0, l)] for k in range(0, l)])
    return np.matmul(x, coeffs)

def dft(x):
    if isinstance(x, list):
        x = np.array(x)
    if x.shape[0] > 1:
        x = dft_col(x)
    if len(x.shape)>1 and x.shape[1] > 1:
        x = dft_row(x)
    return x

def ctdft_col(x):
    l = x.shape[0]
    if l % 2 != 0:
        return dft_col(x)
    evens = ctdft_col(x[0:l:2])
    odds = ctdft_col(x[1:l:2])
    result = np.empty((l, x.shape[1]), dtype = 'complex_')
    for k in range(0, l//2):
        odd = odds[k] * cmath.exp(-2*pi*k*1j/l)
        result[k] = evens[k] + odd
        result[k+l//2] = evens[k] - odd
    return result

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

    
def ctdft(x):
    if isinstance(x, list):
        x = np.array(x)
    if x.shape[0] > 1:
        x = ctdft_col(x)
    if len(x.shape)>1 and x.shape[1] > 1:
        x = ctdft_row(x)
    return x

def ictdft(x):
    f = lambda c: c.real - c.imag*1j
    result = ctdft(f(x))
    return f(result)/x.size

def show_normalized(plt, img, size = None):
    img = abs(img)
    if size != None:
        img = cv2.resize(img, size)
    plot = plt.imshow(img, norm=LogNorm(img.min(), img.max()), cmap="spring")
    fig.colorbar(plot, ax=plt)

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
    orig_size = img.shape[1], img.shape[0]
    size =  2**ceil(log(img.shape[1], 2)), 2**ceil(log(img.shape[0], 2))

    img_scaled = cv2.resize(img, size)
    
    if mode == 1:
        fig, (plt_orig, plt_dft) = plt.subplots(1, 2)
        fig.suptitle("Log scaled |DFT|")
        img_dft = ctdft(img_scaled)
        plt_orig.imshow(img_scaled, cmap="gray")
        show_normalized(plt_dft, img_dft)
    elif mode == 2:
        fig, (plt_orig, plt_denoise) = plt.subplots(1, 2)
        fig.suptitle("Denoising")
        img_dft = denoise(ctdft(img_scaled), 0.8)
        plt_orig.imshow(img, cmap="gray")
        img_denoise = cv2.resize(abs(ictdft(img_dft)), orig_size)
        plt_denoise.imshow(img_denoise, cmap="gray")
    elif mode == 3:
        fig, plts = plt.subplots(2, 3)
        fig.suptitle("Compression")
        img_dft = ctdft(img_scaled)
        threshs = abs(img_dft).flatten()
        threshs.sort()
        for i in (0,1):
            for j in (0,1,2):
                if i==0 and j==0:
                    print(f"Original image: {threshs.size} non-zeroes")
                    plts[i][j].imshow(img, cmap="gray")
                    continue
                ti = round(((0, 0.5, 0.75), (0.9, 0.99, 0.999))[i][j]*threshs.size)
                thresh = threshs[ti]
                print(f"Magnitude threshhold with {threshs.size - ti} non-zeroes; {100*(threshs.size - ti)/threshs.size}% of fourier coefficients remain")
                f = np.vectorize(lambda c: c if abs(c)>=thresh else 0)
                img_dft = f(img_dft)
                img_comp = cv2.resize(abs(ictdft(img_dft)), orig_size)
                plts[i][j].imshow(img_comp, cmap="gray")
        
        
    elif mode == 4:
        rng = np.random.default_rng()
        sizes = [2**p for p in range(5, 11)]
        dft_avg = []
        dft_dev = []
        ctdft_avg = []
        ctdft_dev = []
        for s in sizes:
            dft_runs = []
            ctdft_runs = []
            print(f"Matrix size {s}x{s}")
            for t in range(0,10):
                print(f"\tRun {t+1}")
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
