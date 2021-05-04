from sklearn.preprocessing import PowerTransformer as PT
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import RobustScaler as RS

import numpy as np

def PowerTransform_norm(arr, skip_negs=True):
    negs = np.sum(arr<0,axis = 0)>0
    if not skip_negs:
        arrs[negs,:] -= np.min(arrs[negs,:],axis = 1)
        negs = np.sum(arr<0,axis = 0)>0
    
    scalerPT = PT(method = 'box-cox')
    
    scaler_arr = arr[:,~negs]
    scaler_arr = scaler_arr+0.01*np.min(scaler_arr[scaler_arr>0])
    scalerPT.fit(scaler_arr)
    
    arrPT = arr.copy()
    arrPT[:,~negs] = scalerPT.transform(scaler_arr)
    
    return arrPT

def Standard_norm(arr):
    scalerSS = SS()
    scalerSS.fit(arr)
    arrSS = scalerSS.transform(arr)
    
    return arrSS
    
def Robust_norm(arr):
    scalerRS = RS()
    scalerRS.fit(arr)
    arrRS = scalerRS.transform(arr)
    
    return arrRS