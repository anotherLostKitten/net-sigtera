from fft import *
import numpy as np

if __name__ == "__main__":
    print("Test 1D DFT\n=================================")
    a = [1,2,3,4,5]
    print(a)
    print("naive implementation")
    print(dft(a))
    ctdft_a = ctdft(a)
    print("cooley tukey implementaion")
    print(ctdft_a)
    print("numpy implementation")
    print(np.fft.fft(a))

    print("\nTest 1D inverse DFT (there are rounding errors)\n=================================")
    print(ictdft(ctdft_a))
    
    print("\nTest 2D DFT\n=================================")
    a = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                  [8, 9, 10, 11, 12, 13, 14, 15+1j]])
    print(a)
    print("naive implementaion")
    print(dft(a))
    ctdft_a = ctdft(a)
    print("cooley tukey implementaion")
    print(ctdft_a)
    print("numpy implementation")
    print(np.fft.fft2(a))

    print("\nTest 2D inverse DFT (there are rounding errors)\n=================================")
    print(ictdft(ctdft_a))
    
    
