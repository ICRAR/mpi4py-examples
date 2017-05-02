#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI

from parutils import pprint

comm = MPI.COMM_WORLD

pprint("-"*78)
pprint(" Running on %d cores" % comm.size)
pprint("-"*78)

comm.Barrier()

# Prepare a vector of N=5 elements to be broadcasted...
N = 5
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)    # rank 0 has proper data
else:
    A = np.zeros(N, dtype=np.float64)    # rank 0 has proper data
print("rank {0}: {1}".format(comm.rank, A))
comm.Barrier()
# Broadcast A from rank 0 to everybody
comm.Bcast( [A, MPI.DOUBLE], root=0)

# Everybody should now have the same...
print("[%02d] %s" % (comm.rank, A))
