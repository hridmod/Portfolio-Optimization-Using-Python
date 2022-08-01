from unittest import result
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from numpy.linalg import inv 

covariance_mat=pd.read_excel('Covariance.xlsx',sheet_name='Sheet1')
col_nam=covariance_mat.columns.values.tolist() #convert names of all columns to a list 
col_nam.pop(0) 
temp=".NS" # To download NSE stock data 
col_nam=[i + temp for i in col_nam]
data = yf.download(col_nam,'2022-05-01','2022-06-13')['Adj Close']
daily_returns= data.pct_change().apply(lambda x: np.log(1+x))
E = np.array(daily_returns.mean(axis=0)).reshape(-1,1)
ones = np.ones((E.shape[0],1))
zeros = np.zeros((2,2))

#A@X=b
A = 2*(covariance_mat.iloc[:,1:])
A = np.append(A, E.T, axis=0)
A = np.append(A, ones.T, axis=0)
# A is a 32*30 matrix
temp = np.append(E, ones, axis=1)
temp = np.append(temp, zeros, axis=0)

A = np.append(A, temp, axis=1)
A_inv=inv(A)
# Put together the b vector
b=np.zeros((30,1))
b=np.append(b,(E[0],[1]),axis=0)
result=inv(A)@b

opt_Weight = result[:daily_returns.shape[1]]

print(pd.DataFrame(opt_Weight, index=daily_returns.columns, columns=['Optimal Weights']))


"""OUTPUT
               Optimal Weights
ACC.NS                0.015788
AMBUJACEM.NS         -0.040292
APOLLOTYRE.NS         0.123276
ASHOKLEY.NS           0.057247
ASIANPAINT.NS         0.000041
BERGEPAINT.NS         0.024314
BOSCHLTD.NS           0.017751
BRITANNIA.NS         -0.019475
DABUR.NS              0.068730
EICHERMOT.NS          0.073818
EXIDEIND.NS          -0.026122
GODREJCP.NS          -0.055872
GRASIM.NS             0.103173
HAVELLS.NS           -0.002884
HEROMOTOCO.NS         0.021995
HINDUNILVR.NS         0.001068
ITC.NS               -0.003315
JPASSOCIAT.NS         0.064254
LT.NS                 0.007913
M&M.NS               -0.021257
MARUTI.NS            -0.002718
MOTHERSUMI.NS         0.034197
MRF.NS                0.207407
PIDILITIND.NS        -0.036619
RELINFRA.NS           0.155269
SHREECEM.NS           0.010928
TATACHEM.NS           0.120293
TATAMOTORS.NS        -0.011866
TVSMOTOR.NS           0.023988
ULTRACEMCO.NS         0.088971
The expected portfolio return [-84.43675944]
"""