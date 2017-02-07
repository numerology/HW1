import numpy as np
import pandas as pd
import itertools
from discrete import *

__author__ = 'numerology'

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

df = pd.read_csv('Info_Dataset.csv')

df['d1'] = df['d1'].astype('category').cat.codes
df['d2'] = df['d2'].astype('category').cat.codes
df['d3'] = df['d3'].astype('category').cat.codes
df['d4'] = df['d4'].astype('category').cat.codes
df['d5'] = df['d5'].astype('category').cat.codes
#preprocess

levels = [0,0,0,0,0]
levels[0] = len(list(df['d1'].unique())) 
levels[1] = len(list(df['d2'].unique())) 
levels[2] = len(list(df['d3'].unique())) 
levels[3] = len(list(df['d4'].unique())) 
levels[4] = len(list(df['d5'].unique())) 

# pairwise independence
dm = df.as_matrix().reshape((10000,5))
print(dm.shape[0],dm.shape[1])
# for i1 in range(0, 5):
# 	for i2 in range(i1, 5):
# 		if (i1 == i2):
# 			continue
# 		pair_score = g_square_dis(dm, i1, i2, [], levels)
# 		print('p-value of var:' + str(i1) + ' and var:' + str(i2) + ' is: ' + str(pair_score))

'''
p-value of var:0 and var:1 is: 0.0
p-value of var:0 and var:2 is: 0.231658736846
p-value of var:0 and var:3 is: 0.90331010682
p-value of var:0 and var:4 is: 0.0
p-value of var:1 and var:2 is: 0.166504089458
p-value of var:1 and var:3 is: 0.535798622053
p-value of var:1 and var:4 is: 0.0
p-value of var:2 and var:3 is: 0.0253836754723
p-value of var:2 and var:4 is: 0.804517201563
p-value of var:3 and var:4 is: 0.218212152554

Therefore, using p = 0.05, the only possible pair of independence is (0,1), (0,4), (1,4), (2,3).
When we consider 3-way mutual independence, we only need to consider (0,1,4), other combinations 
are impossible

To verify 3-way independence, we use the following way:
If X independent to Y condition on Z while (X,Y) (Z,Y) are pairwise independent:
P(X)P(Y) = P(X|Z)P(Y|Z) = P(X,Y|Z) = P(X,Y,Z)/P(Z), then P(X,Y,Z) = P(X)P(Y)P(Z)

The only thing we need to verify is that if 1,4 are independent conditioned on 0
'''

pair_score = g_square_dis(dm, 1, 4, set([0]), levels)
print('p-value of var:' + str(1) + ' and var:' + str(4) + ' conditioned on var 0 is: ' + str(pair_score))

'''
p-value is 0.0 for var 1 and var 4 conditioned on var 0
Thus I would argue 0,1,4 are mutually independent
'''

'''
Now explore for more conditional independence
'''
idxAll = range(0,5)

for i1 in range(0, 5):
	for i2 in range(i1, 5):
		if (i1 == i2):
			continue

		sets = []
		for i in idxAll:
			if(i != i1 and i != i2):
				sets.append(i)

		for i in sets:
			pair_score = g_square_dis(dm, i1, i2, set([i]), levels)
			if(pair_score < 0.1):
				print('p-value of var:' + str(i1) + ' and var:' + str(i2) + 
					'conditioned on var: ' + str(i) + ' is: ' + str(pair_score))

		tmpSets = sets
		sets2way = findsubsets(tmpSets, 2)
		for s in sets2way:
			pair_score = g_square_dis(dm, i1, i2, set(s), levels)
			if(pair_score < 0.1):
				print('p-value of var:' + str(i1) + ' and var:' + str(i2) + 
					'conditioned on var (' + str(s[0]) + ',' + str(s[1]) + ')' + ' is: ' + 
					str(pair_score))

		# 3-way conditioning
		pair_score = g_square_dis(dm, i1, i2, set(sets), levels)
		if(pair_score < 0.1):
			print('p-value of var:' + str(i1) + ' and var:' + str(i2) + 
					'conditioned on other vars is: ' + str(pair_score))

'''
Conclusion:
When take significance level p = 0.05:
we have 
p-value of var:1 and var:4 conditioned on var 0 is: 0.0
p-value of var:0 and var:1conditioned on var: 2 is: 0.0
p-value of var:0 and var:1conditioned on var: 3 is: 0.0
p-value of var:0 and var:4conditioned on var: 1 is: 0.0
p-value of var:0 and var:4conditioned on var: 2 is: 0.0
p-value of var:0 and var:4conditioned on var: 3 is: 0.0
p-value of var:1 and var:4conditioned on var: 0 is: 0.0
p-value of var:1 and var:4conditioned on var: 2 is: 0.0

When take significance p = 0.1:
p-value of var:1 and var:4 conditioned on var 0 is: 0.0
p-value of var:0 and var:1conditioned on var: 2 is: 0.0
p-value of var:0 and var:1conditioned on var: 3 is: 0.0
p-value of var:0 and var:4conditioned on var: 1 is: 0.0
p-value of var:0 and var:4conditioned on var: 2 is: 0.0
p-value of var:0 and var:4conditioned on var: 3 is: 0.0
p-value of var:1 and var:2conditioned on var: 3 is: 0.0916463098153
p-value of var:1 and var:4conditioned on var: 0 is: 0.0
p-value of var:1 and var:4conditioned on var: 2 is: 0.0
p-value of var:2 and var:3conditioned on var: 1 is: 0.0551626358929
'''