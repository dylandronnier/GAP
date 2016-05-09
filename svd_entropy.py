#%%
import numpy as np

# For a matrix A, we compute the contribution of every feature using SVD entropy
# E(X)=-1/log(N)*sum(Vj*log(Vj)) with Vj normalized singular values of X
# CE = E_X-E(X), E_X vector of E(X_i) : X_i = X\{X_i}

def entropy(X):
    N=X.shape[0]    
    s=np.linalg.svd(X,compute_uv=False)
    v=s**2
    v=v/np.mean(v)
    E_x=-(1/np.log(N))*np.sum(v*np.log(v))
    return E_x

def featureContribution(X):
    (N,d)=X.shape
    E_x=entropy(X)
    E=np.zeros(d)
    for i in np.arange(d):
        Xi=np.delete(X,i,1)
        E[i]=entropy(Xi)
    return E_x-E

# Select the most interesting features :
def featureSelection(X,method,r0=1):
# - X : data matrix
# - method : see SVD paper sent by Sami, page 123-124
# "partition" : first method, page 123
# "SR","FS1","FS2","BE" : other methods, page 124
    if method=="partition":
        CE=featureContribution(X)
        c=np.mean(CE)
        s=np.std(CE)
        high_contribution=X[:,CE>c+s]
        low_contribution=X[:,CE<c-s]
        average_contribution=X[:,~((CE>c+s) | (CE<c-s))]
        return (high_contribution,average_contribution,low_contribution)
    elif method=="SR":
        CE=featureContribution(X)
        featuresSelected=np.argsort(CE)
        X=X[:,featuresSelected]
        return X[:,-r0]
    elif method=="FS1":
        #initialisation : we chose the feature with highest CE
        (N,d)=X.shape
        CE=featureContribution(X)
        featuresSelected=np.zeros(d,dtype=bool)
        featuresSelected[np.argmax(CE)]=True
        # loop
        for i in np.arange(1,r0):
            featuresNotSelected=np.where(featuresSelected==0)
            entropies=np.zeros(N-i)
            for j in featuresNotSelected:
                featuresSelected[j]=True
                entropies[j]=entropy(X[:,featuresSelected])
                featuresSelected[j]=False
            featuresSelected[np.argmax(entropies)]=True
        return X[:,featuresSelected]
    elif method=="FS2":
        N=X.shape[0]
        newX=np.zeros((N,r0))        
        for i in np.arange(r0): 
           CE=featureContribution(X)
           maxFeature=np.argmax(CE)
           X=np.delete(X,maxFeature,1)
           newX[:,i]=X[:,i]
        return X
    elif method=="BE":
       for i in np.arange(r0):
           CE=featureContribution(X)
           minFeature=np.argmin(CE)
           X=np.delete(X,minFeature,1)
       return X
    else:
        print("Wrong method argument : try partition,SR,FS1,FS2 or BR")


if __name__=='__main__':
    A = np.random.random((2000,5))

      
