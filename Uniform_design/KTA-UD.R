library(gtools)
MD <- function(D){
  n <- dim(D)[1]
  m <- dim(D)[2]
  sum1 <- 0
  for(i in 1:n){
    for(j in 1:n){
      sum2 <- 1
      for(k in 1:m){
        sum2 <- sum2*(15/8-0.25*abs(D[i,k])-0.25*abs(D[j,k])-0.75*abs(D[i,k]-D[j,k])+0.5*(D[i,k]-D[j,k])^2)
      }
      sum1 <- sum1+sum2
    }
  }
  sum3 <- 0
  for(i in 1:n){
    sum4 <- 1
    for(k in 1:m){
      sum4 <- sum4*(5/3-0.25*abs(D[i,k])-0.25*D[i,k]^2)
    }
    sum3 <- sum3+sum4
  }
  return((19/12)^m-2*sum3/n+sum1/(n^2))
}
f <- function(a,b,beta){
  l <- length(a)
  sum1 <- 1
  for(i in 1:l){
    sum1 <- sum1*(15/8-0.25*abs(a[i])-0.25*abs(b[i])-0.75*abs(a[i]-b[i])+0.5*(a[i]-b[i])^2)/beta
  }
  return(sum1)
}
MDD <- function(D,t,beta){
  n <- dim(D)[1]
  m <- dim(D)[2]
  sum1 <- 0
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      sum1 <- sum1 + ((f(D[i,], D[j,], beta)+f(D[i,], -D[j,], beta))/2)^t
    }
  }
  sum1 <- 2*sum1
  for(i in 1:n){
    sum1 <- sum1 + ((f(D[i,], D[i,], beta)+f(D[i,], -D[i,], beta))/2)^t
  }
  sum3 <- 0
  for(i in 1:n){
    sum4 <- 1
    for(k in 1:m){
      sum4 <- sum4*((5/3-0.25*abs(D[i,k])-0.25*D[i,k]^2)/beta)^t
    }
    sum3 <- sum3+sum4
  }
  return(-2*sum3/n+sum1/(n^2))
}
ex <- function(D, vs){
  N <- dim(D)[1]
  m <- dim(D)[2]
  r <- sample(c(1:m),2)
  n1 <- sample(c(1:N),1)
  n2 <- sample(c(1:N),1)
  Dnew <- D
  Dnew[n1,r[1]] <- sample(vs[which(vs!=D[n1,r[1]])],1)
  Dnew[n2,r[2]] <- sample(vs[which(vs!=D[n2,r[2]])],1)
  return(Dnew)
}
generate_UD <- function(N, m, t, I = 10, J = 1000, alpha = 0.1, beta = 1.75){
  s <- ceiling(N/2^t)
  Rrow <- s
  Rcol <- ceiling(m/t)
  Hrow <- 2^t
  Hcol <- t
  print(paste0("the number of row for design D is :", Rrow, ",the number of column for design D is:", Rcol))
  print(paste0("the number of row for design H is :", Hrow, ",the number of column for design H is:", Hcol))
  H <- permutations(2,t,c(-1,1),repeats.allowed = T)
  vs <- (2*c(0:(s-1))-(s-1))/(2*s)
  R <- matrix(0, s, ceiling(m/t))
  v <- vector()
  minn <- Inf
  for(tt in 1:100){
    for(i in 1:Rcol){
      R[,i] <- sample(vs,s)
    }
    a <- MDD(R,t,beta)
    if(a<minn){
      minn <- a
      D <- R
    }
    v <- append(v,a)
  }
  tau <- alpha*min(v)
  print(paste0("tau=", tau))
  i <- 1  
  rmin <- Inf
  while (i<=I) {
    print(paste0("Starting the ", i, "th round of optimization"))
    for(j in 1:J){
      D_new <- ex(D, vs)
      a <- MDD(D_new,t,beta)
      if(a<rmin){
        rmin <- a
        D <- D_new
        print(paste0("Optimization successful, current criterion value is ", rmin))
      }
      if(a<rmin+tau){
        D <- D_new
      }
      if(j%%100==0){
        print(paste0("This round has completed ", j, "/", J))
      }
    }
    tau <- (I-i)*tau/I
    i <- i+1
  }
  DD <- kronecker(H,D)
  D_final <- DD[sample(c(1:dim(DD)[1]), N), sample(c(1:dim(DD)[2]),m)]
  D_final <- D_final + 0.5
  uniform_matrix <- matrix(runif(N*m, min=-0.5/s, max=0.5/s), nrow=N, ncol=m)
  DD <- D_final + uniform_matrix
  return(DD)
}

Q <- generate_UD(20000,128,6,10,1000,0.1,1.75)