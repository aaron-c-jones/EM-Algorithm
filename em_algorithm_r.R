#reading in data
missing_data=read.csv('/Volumes/AARON/Datasets/missing_data_test.csv',header=FALSE,strip.white=TRUE)
missing_data=as.matrix(missing_data)

em<-function(data,max_it=1000,tol_err=1e-12){
  p<-ncol(data)
  
  blank_row<-which(rowSums(is.na(data))==p)
  if(length(blank_row)>0){data<-data[-blank_row,]}
  
  #initializing algorithm values
  n<-nrow(data)
  mod_rel_err<-it<-1
  
  #initializing mu and sigma
  mu_init<-apply(data,2,mean,na.rm=T)
  pred<-data
  for(i in 1:p){
    pred[is.na(data[,i]),i]<-mu_init[i]
  }
  sig_init<-(n-1)/n*cov(pred) #biased cov
  
  while(mod_rel_err>tol_err&it<=max_it){
    
    #initialize cov structure
    temp_m<-temp_s<-0
    
    #initialize theta
    theta<-c(mu_init,sig_init)
    
    for(i in 1:n){
      x_st<-matrix(as.numeric(data[i,]),ncol=1)
      
      if(sum(is.na(x_st))!=0){
        pos<-which(is.na(x_st))
        x_st[pos,]<-mu_init[pos]+sig_init[pos,-pos]%*%
          solve(sig_init[-pos,-pos])%*%
          matrix(x_st[-pos,]-mu_init[-pos],ncol=1)
        pred[i,]<-t(x_st)
      }
      temp_m<-temp_m+x_st
    }
    mu_new<-temp_m/n
    
    for(i in 1:n){
      x_st<-matrix(as.numeric(data[i,]),ncol=1)
      s_st<-x_st%*%t(x_st)
      pred_i<-pred[i,]
      
      if(sum(is.na(x_st))!=0){
        pos<-which(is.na(x_st))
        s_st[pos,pos]<-sig_init[pos,pos]-sig_init[pos,-pos]%*%
          solve(sig_init[-pos,-pos])%*%sig_init[-pos,pos]+
          pred_i[pos]%*%matrix(pred_i[pos],nrow=1)
        s_st[-pos,pos]<-x_st[-pos,]%*%matrix(pred_i[pos],nrow=1)
        s_st[pos,-pos]<-t(s_st[-pos,pos])
      }
      temp_s<-temp_s+s_st
    }
    sig_new<-temp_s/n-mu_new%*%t(mu_new)
    theta_new<-c(mu_new,sig_new)
    mod_rel_err<-norm(matrix(theta_new-theta),type='f')/
      max(1,norm(matrix(theta)))
    
    #print(mu_new)
    #print(sig_new)
    
    it<-it+1
    mu_init<-mu_new
    sig_init<-sig_new
  }
  return(list(mu_hat=mu_new,
              sig_hat=sig_new,
              imputed_data=pred,
              iteration=it-1))
}
result=em(missing_data)









