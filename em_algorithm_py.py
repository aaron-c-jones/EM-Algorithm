#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:39:44 2017

@author: aaronjones
"""

import numpy as np

np.random.seed(1)

mu=np.array([3.,-2.,1.])
sigma=np.array([[10.,5.,4.],[5.,18.,7.],[4.,7.,9.]])
#mu=np.array([1.,-4.,2.5])
#sigma=np.array([[1.,0.7,0.7],[0.7,2.,0.7],[0.7,0.7,1.5]])
data=np.random.multivariate_normal(mean=mu,cov=sigma,size=500)
index=np.array([np.random.choice(a=[1,np.nan],size=500,p=[0.8,0.2]),
                np.random.choice(a=[1,np.nan],size=500,p=[0.8,0.2]),
                np.random.choice(a=[1,np.nan],size=500,p=[0.8,0.2])]).T
md=np.multiply(data,index)

#np.savetxt("missing_data_test.csv",md,delimiter=",")

def em_algorithm(md,max_it=5,tol_err=1e-12):
    md=md[~np.isnan(md).all(axis=1)]
    
    n,p=md.shape
    mod_rel_err=it=1.0
    
    mu_init=np.nanmean(md,axis=0)
    pred=md.copy()
    for i in range(p):
        pred[np.isnan(md[:,i]),i]=mu_init[i]
    sig_init=(n-1)/float(n) * np.cov(pred.T)
    
    sig_init_reshape=sig_init.reshape(1,sig_init.size)
    theta_init=np.append(mu_init,sig_init_reshape).reshape(sig_init_reshape.size+mu_init.size,1)

    while mod_rel_err>tol_err and it<=max_it:
        temp_m=temp_s=0

        for i in range(n):
            x_st=md[i,:].reshape(p,1).copy()
            
            if np.isnan(x_st).sum()!=0:
                pos=np.argwhere(np.isnan(x_st))[:,0]
                x_st[pos,:]=mu_init[pos].reshape(len(pos),1)+np.dot(np.dot(np.delete(sig_init[pos,:],pos,1),np.linalg.inv(np.delete(np.delete(sig_init,pos,0),pos,1))),np.delete(x_st,pos,0)-np.delete(mu_init.reshape(p,1),pos,0))
                pred[i,:]=x_st.T
            temp_m=temp_m+x_st
        mu_new=temp_m/float(n)
        
        for i in range(n):
            x_st=md[i,:].reshape(p,1)
            s_st=np.dot(x_st,x_st.T)
            pred_i=pred[i,:]
            
            if np.isnan(x_st).sum()!=0:
                pos=np.argwhere(np.isnan(x_st))[:,0]
                s_st[pos[:,None],pos]=sig_init[pos[:,None],pos]-np.dot(np.dot(np.delete(sig_init[pos,:],pos,1),np.linalg.inv(np.delete(np.delete(sig_init,pos,0),pos,1))),np.delete(sig_init[:,pos],pos,0))+np.dot(pred_i[pos].reshape(len(pred_i[pos]),1),pred_i[pos].reshape(1,len(pred_i[pos])))
                s_st[np.delete(np.arange(p),pos,0)[:,None],pos]=np.dot(np.delete(x_st,pos,0),pred_i[pos].reshape(1,len(pred_i[pos])))
                s_st[pos,np.delete(np.arange(p),pos,0)[None,:]]=s_st[np.delete(np.arange(p),pos,0),pos]
            temp_s=temp_s+s_st
        sig_new=temp_s/float(n)-np.dot(mu_new,mu_new.T)
        
        sig_new_reshape=sig_new.reshape(1,sig_new.size)
        theta_new=np.append(mu_new,sig_new_reshape).reshape(sig_new_reshape.size+mu_new.size,1)
        
        print(mu_new)
        print(sig_new)
        
        mod_rel_err=np.linalg.norm(theta_new-theta_init)/max([1,np.linalg.norm(theta_init)])
        
        it=it+1
        
        mu_init=mu_new.copy()
        sig_init=sig_new.copy()
        theta_init=theta_new.copy()
    
    return mu_new,sig_new,pred,it-1

mu_new,sig_new,pred,it=em_algorithm(md)

        












    

