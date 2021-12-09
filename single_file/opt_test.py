import sys
import numpy as np

x=float(sys.argv[3]); y=float(sys.argv[4])
s=float(sys.argv[1])
np.random.seed( int(int(sys.argv[2])*1234*(x+5)*(y+5)) ) 
mu = -((x)**2 + (2*(y))**2)
print (np.random.normal(mu, s, 1)[0])