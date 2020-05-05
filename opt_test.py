
import sys
import numpy as np

x=float(sys.argv[2]); y=float(sys.argv[3])
np.random.seed( 100000+int(sys.argv[1])*int((x+2)*(y+2)**2) ) 
mu = (50 - x**2 - (2*y)**2)/50.0
print (np.random.normal(mu, 0.05, 1)[0])