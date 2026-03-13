# Experiments on vessel and diabetes dataset
library(UniDOE)
library(SPlit)
library(ggplot2)
library(numbers)
library(knn.covertree)
library(h2o)
h2o.init()
# Utility function
## max_min is used to normalize the input data in matrix form (N,d) into [0,1]^d by max-min normalization.
max_min=function(x){
  n=nrow(x)
  d=ncol(x)
  y=matrix(0,n,d)
  for(i in 1:d){
    M=max(x[,i])
    m=min(x[,i])
    y[,i]=(x[,i]-m) / (M-m)
  }
  return(y)
}

## Below three functions are used for generating design.
## GLP is used to generate design in terms of different discrepancy. We recommend using 'WD2' to generate design for better performance under high dimension cases.
## BRUD function is used to perform BRUD transformation which obtain several randomly perturbed nearly optimal designs given fixed design.
## CVgroup is used to run divide-and-conquer strategy on large-scale datasets.
GLP<-function(n,p,type="CD2"){
  fb<-c(3,5,8,13,21,34,55,89,144,233,377,610,987)
  if(((n+1)%in%fb)&(p==2)){
    design0<-matrix(0,(n+1),p)
    H<-rep(1,2)
    H[2]<-fb[which(fb==(n+1))-1]
    for (j in 1:p) {
      for (i in 1:(n+1)) {
        design0[i,j]<-(2*i*H[j]-1)/(2*(n+1))-floor((2*i*H[j]-1)/(2*(n+1)))
      }
    }
    design0<-design0[-(n+1),]*(n+1)/n
  }else{
    if(p==1){
      design0<-matrix(0,n,p)
      for(i in 1:n){
        design0[i,1]<-(2*i-1)/(2*n)
      }
      return(design0)
    }
    h<-c()
    for(i in 2:min((n+1),200)){
      if(coprime((n+1),i)==T){
        h<-c(h,i)
      }
    }
    if(p>2){
      for (i in 1:100) {
        if(choose(p+i,i)>5000){
          addnumber<-i
          break
        }
      }
      h<-h[sample(1:length(h),min(length(h),(p+addnumber)))]
    }
    H<-combn(h,p,simplify = F)
    if(length(H)>3000){
      H<-H[sample(3000)]
    }
    design0<-matrix(0,n,p)
    d0<-DesignEval(design0,crit=type)
    for (t in 1:length(H)) {
      design<-matrix(0,n,p)
      for (i in 1:p) {
        for (j in 1:n) {
          design[j,i]<-(j*H[[t]][i])%%(n+1)
        }
      }
      d1<-DesignEval(design,crit=type)
      if(d1<d0){
        d0<-d1
        design0<-design
      }
    }
    design0<-(design0*2-1)/(2*n)
  }
  return(design0)
}
BRUD<-function(Design){
  D=Design
  n=nrow(D)
  s=ncol(D)
  rand=matrix(runif((n*s),0,1),nrow=n,ncol=s)
  eta_mat=matrix(rep(sample(c(0:(n-1)),s,replace = TRUE),n),nrow=n,ncol=s,byrow=TRUE)
  eta_mat=((eta_mat-0.5)/n)
  RUD=((D+eta_mat)%%1+rand/n)%%1
  #RUD=1-abs(2*RUD-1)
  return(RUD)
}
CVgroup <- function(k,datasize){
  cvlist <- list()
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    
  temp <- sample(n,datasize)   
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x,function(x) dataseq[temp==x])  
}

## Given dataset in matrix form (N,d), select uniform subsample with size $n*k$.
## It is worth mentioning that since one-nearest-neighbor approximation may find same neighbor for different input data points. Set $n$ little larger could results in a subsample with desired size. 
dnn=function(data,n,k,design=NULL){
  indices=array()
  data_collection=list()
  un_sampled_collection=list()
  N=dim(data)[1]
  d=dim(data)[2]
  indices_collection=CVgroup(k,N)
  if(is.null(design)){
    design=GLP(n,d,"WD2")
  }
  else{
    if(n!=nrow(design) || d!=ncol(design)){
      stop("The subsetsize and dimension should match the run size and dimension of given uniform design")
    }
  }
  for(i in 1:k){
    data_collection[[i]]=data[indices_collection[[i]],]
    ud=BRUD(design)
    kdtree<-nn2(data=data_collection[[i]],query=ud,k=1,treetype="kd")
    un_sampled_collection[[i]]=unique(kdtree$nn.idx)
    sampled_indices=indices_collection[[i]][un_sampled_collection[[i]]]
    indices=c(indices,sampled_indices)
  }
  indices=indices[-1]
  return(indices)
}

### Running code for experiments on diabetes and vessel.

## Load training dataset Diabetes (see Vessel in later section).

train_diabete=read.csv("diabetes_binary_health_indicators_BRFSS2015.csv")
test_diabetes=read.csv("diabetes_binary_health_indicators_BRFSS2021.csv")

# Max-min scaling
train_diabetes=max_min(train_diabetes)
test_diabetes=max_min(test_diabetes)
# Load testing dataset in h2o form. For classification problem, the column indicating the outcome should be converted into factor so that h2o.gbm would function properly.
test_diabetes=as.h2o(test_diabetes)
test_diabetes$V1=as.factor(test_diabetes$V1)

# total size of subsample
N=30000
# matrix for saving the results
robust_matrix=matrix(0,100,4)
# generate design before running repeatedly. This could save a lot computation time.
uddnn=GLP(500,21,"WD2")
all=c(1:nrow(train_diabetes))
for(i in 1:100){
  # The first column indicating the outcome and hence should be excluded when selecting uniform subsample
  uniform_indices=dnn(train_diabetes[,-1],500,80,uddnn)
  # Robust subsample consists of two parts. One is uniform subsample whose indices is obtained by dnn. The other is random subsample which is selected randomly from the rest.
  rest_indices=sample(all[-uniform_indices],N-length(uniform_indices))
  # The total size should match $N$. train gives the robust subsample selected by SimSRT.
  robust_train2=c(uniform_indices,rest_indices)
  train=as.h2o(train_diabetes[robust_train2,])
  train$V1=as.factor(train$V1)
  model=h2o.gbm(2:22,1,train)
  performance=h2o.performance(model,test_diabetes)
  robust_matrix[i,]=as.matrix(h2o.metric(performance,0.5)[[1]])[1,c(1,4,9,10)]
}


### Example of vessel fitting.

## Load training datasets vessel
vessel_training=read.csv('train.csv')
vessel_testing=read.csv('dev_out.csv')

## The eleven column represent time_id for observation. Hence this column is excluded for prediction.
vessel_training=vessel_training[,-11]
vessel_testing=vessel_testing[,-11]
## The last column indicates the true consumed energy (the quantity we predict). Therefor only covariates that are responsible for predictions are max-min scaled.
vessel_training[,-11]=max_min(vessel_training[,-11])
vessel_testing[,-11]=max_min(vessel_testing[,-11])
## The continuous responses does not need to be converted into factor form before feeding into h2o.xgboost.

# Total size of subsample
n=5000
# Save the performance result
robust_matrix=matrix(0,100,2)
all=c(1:nrow(vessel_training))
uddnn=GLP(500,10,"WD2")
for(i in 1:100){
  uniform_indices=dnn(vessel_training[,-11],500,9,uddnn)
  rest_indices=sample(all[-uniform_indices],n-length(uniform_indices))
  robust_train2=c(uniform_indices,rest_indices)
  train=as.h2o(vessel_training[robust_train2,])
  model=h2o.gbm(1:10,11,train)
  performance=h2o.performance(model,vessel_testing)
  robust_matrix[i,1]=performance@metrics$RMSE
  robust_matrix[i,2]=performance@metrics$mae
}




