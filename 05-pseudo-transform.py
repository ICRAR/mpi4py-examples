#!/usr/bin/env python

"""

How to run:

   mpirun -np <NUM> ./pseudo-whitening <INPUT-CUBE.fits> <OUTPUT-CUBE.fits>

"""

from __future__ import division

import sys
import pyfits
import numpy as np
from numpy.fft import fft2, ifft2
from mpi4py import MPI
#from bernstein.utils import autotable
from parutils import pprint

#=============================================================================
# Main
comm = MPI.COMM_WORLD
in_fname = sys.argv[-2]
out_fname = sys.argv[-1]

#in_fname = 'L1448_13CO.fits.gz'
left = 145
right = 245
rem = (right - left) % comm.size
if (rem != 0):
    right += comm.size - rem

images = pyfits.getdata(in_fname)[left:right, :, :]#h5in.root.images
image_count, height, width = images.shape
image_count = min(image_count, 200)

# rem = image_count % comm.size
# if (image_count % comm.size != 0):
#     extra_col = comm.size - rem
#     pprint("image_count % comm.size != 0")
#     sys.exit(1)

pprint("============================================================================")
pprint(" Running %d parallel MPI processes" % comm.size)
pprint(" Reading images from '%s'" % in_fname)
pprint(" Processing %d images of size %d x %d" % (image_count, width, height))
pprint(" Writing transformed images into '%s'" % out_fname)

# Prepare convolution kernel in frequency space
kernel_ = np.ones((height, width))

# rank 0 needs buffer space to gather data
if comm.rank == 0:
    gbuf = np.empty( (comm.size, height, width) )
    origin_header = pyfits.open(in_fname)[0].header
    new_images = np.zeros((image_count, height, width))
else:
    gbuf = None



# Distribute workload so that each MPI process processes image number i, where
#  i % comm.size == comm.rank.
#
# For example if comm.size == 4:
#   rank 0: 0, 4, 8, ...
#   rank 1: 1, 5, 9, ...
#   rank 2: 2, 6, 10, ...
#   rank 3: 3, 7, 11, ...
#
# Each process reads the image from the FITS cube file by itself. Sadly, FITS
# does not support parallel writes from multiple processes into the same fits
# file. So we have to serialize the write operation: Process 0 gathers all
# transformed images and writes them.

comm.Barrier()                    ### Start stopwatch ###
t_start = MPI.Wtime()

for i_base in range(0, image_count, comm.size):
    i = i_base + comm.rank
    #
    if i < image_count:
        img  = np.nan_to_num(images[i])            # load image from HDF file
        img_ = fft2(img)            # 2D FFT
        whi_ = img_ * kernel_       # multiply with kernel in freq.-space
        #whi_  = img_
        whi  = np.abs(ifft2(whi_))#.astype(np.float)  # inverse FFT back into image space

    # rank 0 gathers transformed images
    comm.Gather(
        whi,   # send buffer
        gbuf,  # receive buffer
        root=0               # rank 0 is root the root-porcess
    )

    # rank 0 has to write into the HDF file
    if comm.rank == 0:
        # Sequentially append each of the images
        for r in range(comm.size):
            new_images[i + r, :, :] = gbuf[r]

if comm.rank == 0:
    hdu = pyfits.PrimaryHDU(new_images)
    hdu.header = origin_header
    hdu.writeto(out_fname)

comm.Barrier()
t_diff = MPI.Wtime()-t_start      ### Stop stopwatch ###

#h5in.close()
#h5out.close()

pprint(
    " Transformed %d images in %5.2f seconds: %4.2f images per second" %
        (image_count, t_diff, image_count/t_diff)
)
pprint("============================================================================")
