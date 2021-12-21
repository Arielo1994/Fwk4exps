import sys
import numpy as np
import hashlib  

x=float(sys.argv[3]); y=float(sys.argv[4])
s=float(sys.argv[1])

str= sys.argv[1]+sys.argv[2]+sys.argv[3]+sys.argv[4]
hash=int(hashlib.sha1(str.encode("utf-8")).hexdigest(), 16) % (2 ** 32) 
np.random.seed(hash) 
mu = -(x**2 + 2*y**2)
print (np.random.normal(mu, s, 1)[0])