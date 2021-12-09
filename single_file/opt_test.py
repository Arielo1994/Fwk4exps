import sys
import numpy as np

x=float(sys.argv[3]); y=float(sys.argv[4])
s=float(sys.argv[1])
np.random.seed( int(sys.argv[2])*1234 ) 
mu = -((x-1)**2 + (2*(y-1))**2)
print (np.random.normal(mu, s, 1)[0])