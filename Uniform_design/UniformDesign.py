#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
from numpy import *
import pandas as pd
import itertools
import time
import math
import random
import matplotlib.pyplot as plt
import numpy.matlib
from scipy import stats
from decimal import Decimal, getcontext
import warnings

getcontext().prec = 10000  # 设置精度足够高
warnings.filterwarnings("ignore")
from IPython.display import display, HTML
display(HTML('<style>.container{width:90% !important;}</style>'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


def FullFFD(s,m): #全设计s**m行，s水平，m列
    matrix=[]
    target=[]
    for i in range(s):
        target.append(i)
    for item in itertools.product(target,repeat=m):
        numbers = list(map(int, item))
        matrix.append(numbers)
    return np.array(matrix)  

def Prime(x): #判断一个数是否是素数，是则返回1，不是返回0
    if x == 2: #若x是2
        return 1
    elif x % 2 == 0: #若x是2的倍数
        return 0
    for i in range(3, int(x ** 0.5) + 1, 2): #遍历3到sqrt(x)之间的奇数
        if x % i == 0:
            return 0
    return 1

def Map_to_Cube(D0): #将一个s水平的设计D0映射到单位立方体上
    (N,m)=shape(D0)
    s=len(np.unique(D0[:,0]))
    if min(D0[:,0])==0: #若D0的第一列中有0元，则显然D0水平为0,...,s-1
        D=D0+1 #将水平从{0,...,s-1}变为{1,...,s}
    else: #若D中没有0元，其水平为1,...,s
        D=D0    
    return (2 * D - 1)/(2*s)

def Collapse(D0,s): #将一个s0水平设计D0向s水平塌陷
    (N,m)=shape(D0)
    s0=len(np.unique(D0[:,0]))
    if min(D0[:,0])==0: #若D0的第一列中有0元，则显然D0水平为0,1,...,s-1
        D=D0
    else: #若D中没有0元，其水平为1,...,s
        D=D0-1 #将水平从{1,...,s}变为{0,...,s-1}
    if s==s0:
        DD=D
    else:
        L=s0/s
        DD=np.floor(D/L)
    return DD

def DeleteEC(D0, s): #删除设计D0中在水平s意义下的等价列，要求D0本身无重复列且水平数从0开始
    DD=np.array(D0).T
    (N,n)=shape(DD)
    L1=[]
    L2=[]
    for i in range(N):
        Show=0
        g=DD[i]
        gEC=mod(np.zeros(n)-g, s) #取g的等价列
        for j in range(i):
            if np.array_equal(DD[j], gEC): #若存在等价列
                Show=Show+1
                break
            else:
                Show=Show+0
        if Show==0:
            L1.append(i)
        else:
            L2.append(i)
    D1=D0[:,L1]
    D2=D0[:,L2]
    return (D1,D2)

def WD_fast(D0): #计算一个设计的平方WD值
    (N,n)=shape(D0)
    if n==0:
        wd=0
    else:
        s=np.array(np.max(D0, axis=0)) 
        if max(s) >= 1:
            if min(np.min(D0,axis=0))==0: #若D0中有0元，其水平为0,1,...,s-1
                s=s+1 #D0每一列的水平数
                t=mat(np.ones([N,1])) * mat(s)
                D=(D0+0.5)/t
            else: #若D0中没有1元，其水平为1，2,...,s
                t=mat(np.ones([N,1])) * mat(s)
                D=(D0-0.5)/t
        else: #若D0是单位立方体上
            t=mat(np.ones([N,1])) * mat(s)
            D=D0
        LLL=[]
        for i in range(N-1): #遍历前N-1行
            X1=mat(np.ones([N-i-1,1])) * mat(D[i,:])
            A=np.array(X1-D[range(i+1,N),:])
            LLL.append(sum(np.prod( 3/2 - abs(A) + A**2, axis=1 )))
        summary=2*sum(LLL)
        wd=-(Decimal(4) / Decimal(3))**n + ((Decimal(3) / Decimal(2))**n)/Decimal(N) + Decimal(summary)/Decimal(N**2)  
    return float(wd)

def LB_WD(N,n,s): #平方WD值的下界
    if n==0:
        return 0
    else:
        lamb_1=(N-s)*(n)/(s*(N-1))
        lamb_2=N*n/(s*(N-1))
        a = float(-(Decimal(4) / Decimal(3))**n + ((Decimal(3) / Decimal(2))**n)/Decimal(N))
        p=1
        if s%2 != 0: # 若s为奇
            for i in range(1,int((s+1)/2)):
                x=3/2-(i*(s-i))/(s**2)
                p=p*x
            b = (N-1)/(N) * ((3 / 2)**(lamb_1)) * (p**(2*(lamb_2)))
        else: # 若s为偶
            for i in range(1,int(s/2)):
                x=3/2-(i*(s-i))/(s**2)
                p=p*x
            b = (N-1)/(N) * ((3 / 2)**(lamb_1)) * ((5 / 4)**(lamb_2)) * (p**(2*(lamb_2)))
        LB=a+b
        return max(0,LB)

def Resort(D0): #在切割之后对每一列中的各个元素进行大小重排
    (N,n)=shape(D0)
    D=np.zeros([N,n])
    for j in range(0,n):
        Sort=np.argsort(D0[:,j])
        for i in range(0,len(Sort)):
            D[Sort[i],j]=i
    return D

def PowerGen(N): #用方幂好格子点法选出生成向量
    I=[]
    L=[]
    LENGTH=[]
    for i in range(2,N):
        if gcd(i,N)==1:
            J=[]
            a=1
            for j in range(1,N):
                a=int(mod(a*i, N))
                J.append(a)
            J=list(set(list(J)))
            length=len(J)
            if length not in LENGTH:
                I.append(i) #记录所有可以用的生成元
                LENGTH.append(length) #记录每个生成元所得的生成向量的长度
                L.append(J) #记录这些生成向量
    return (I, LENGTH, L)


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


def Cutting(N, n, s, T, D0 = []): #从一个已知设计D0或一个能覆盖参数的均匀LHD中切割出所需设计，T为最大切割次数，即迭代次数
    if n == 0: 
        wd=0
        D=np.zeros([N,0])
        return (wd, D)
    elif D0 == []: #若没有定义初始设计，找一个规模更大的LHD 
        p=N #找一个大于等于N的最小素数p，显然p是奇素数
        while Prime(p)==0: 
            p=p+1  
        X=FullFFD(p,1)
        G=numpy.delete(np.array(mat(X.T)),0,axis=1)    
        XG=mod(np.array(mat(X)*mat(G)),p) #XG为p行p-1列的均匀LHD
        k=int(n/(p-1)) #需要k个XG并置
        n1=int(n-k*(p-1)) #在并置基础上还需n1列, n1<p-1
        D00=np.zeros([p,0])
        for i in range(0, k): #对XG随机行置换后再并置
            D00=np.concatenate((D00, XG[random.sample(range(p), p),:]), axis=1) #此时D00有k*(p-1)列        
        if n1==0:
            D0=D00
        elif n1 > int((p-1)/2): #再并置一个XG
            D0=np.concatenate((D00, XG[random.sample(range(p), p),:]), axis=1) 
        else: #取XG的一半并置上去
            GG=DeleteEC(G,p)[0]
            XGG=mod(np.array(mat(X)*mat(GG)),p) 
            D0=np.concatenate((D00, XGG[random.sample(range(p), p),:]), axis=1) 
        n0=shape(D0)[1]         
        wd=float("inf")  
        for i in range(0,p-N+1):
            for j in range(0,n0-n+1):
                if (i+1)*(j+1) <= T: #限制迭代次数
                    #D0的一个子集先水平重新排序再水平塌陷
                    dd = Collapse(Resort(D0[range(i,i+N),:][:,range(j,j+n)]), s)
                    wd_dd=WD_fast(dd)
                    if wd_dd < wd:
                        wd=wd_dd
                        D=dd
        return (wd, D, p) #返回推荐的行数p              
    else: #若定义了初始设计D0，直接切割即可       
        (N0,n0)=shape(D0)
        wd=float("inf")   
        for i in range(0,N0-N+1):
            for j in range(0,n0-n+1):
                if (i+1)*(j+1) <= T: #限制迭代次数
                    #D0的一个子集先水平重新排序再水平塌陷 
                    dd = Collapse(Resort(D0[range(i,i+N),:][:,range(j,j+n)]), s)
                    wd_dd=WD_fast(dd)
                    if wd_dd < wd:
                        wd=wd_dd
                        D=dd               
        return (wd, D) 

def GeneratorMartix(N, n, s, T): #从生成阵里切割出n列；若列数较多就列并置几个（要求s是素数，N是s的幂）
    if s==2: #若s是偶素数
        Pr_even=1
    else: #若s是奇素数
        Pr_odd=1
        for i in range(1,int((s+1)/2)):
            x=3/2-(i*(s-i))/(s**2)
            Pr_odd=Pr_odd*x  
    #根据N和s生成满足行差条件的设计XG——————————————————————————————
    m=round(math.log(N,s)) #生成矩阵G的行数m
    X=FullFFD(s,m)
    G=numpy.delete(np.array(mat(X.T)),0,axis=1)
    wd=float("inf")
    a = float(-(Decimal(4) / Decimal(3))**n + ((Decimal(3) / Decimal(2))**n)/Decimal(N))
    #根据水平数构造XG--------——————————————————————————————————
    if s==2: #若s是偶素数  
        XG=mod(np.array(mat(X)*mat(G)),s)
        ng=N-1 #XG的列数
        k=int(n/ng) #需要k个XG并置
        n1=int(n-k*ng) #在并置基础上还需n1列, n1<N-1
        D0=np.zeros([N,0])
        for i in range(0, k): #对XG随机行置换后再并置
            D0=np.concatenate((D0, XG[random.sample(range(N), N),:]), axis=1) #此时D0有k*ng=n-n1列 
        lamb_1=(N-s)*(k*ng)/(s*(N-1))
        lamb_2=N*(k*ng)/(s*(N-1))
        KK=float((Decimal(3)/Decimal(2))**Decimal(lamb_1) * (Decimal(5)/Decimal(4))**Decimal(lamb_2) * (Decimal(Pr_even)**(Decimal(2)*Decimal(lamb_2))))
    else: #若s是奇素数,将XG的一半视为一个单元,仍记为XG
        GG=DeleteEC(G,s)[0]
        XG=mod(np.array(mat(X)*mat(G)),s)
        XGG=mod(np.array(mat(X)*mat(GG)),s)
        ng=int(N-1)
        ngg=int((N-1)/2) #XGG的列数
        k=int(n/ng) #需要k个XG并置
        n1=int(n-k*ng) #在并置基础上还需n1列, n1<(N-1)
        D0=np.zeros([N,0])
        for i in range(0, k): #对XG随机行置换后再并置
            D0=np.concatenate((D0, XG[random.sample(range(N), N),:]), axis=1) #此时D0有k*ng=n-n1列 
        lamb_1=(N-s)*(k*ng)/(s*(N-1))
        lamb_2=N*(k*ng)/(s*(N-1))
        KK=float((Decimal(3)/Decimal(2))**Decimal(lamb_1) * (Decimal(Pr_odd)**(Decimal(2)*Decimal(lamb_2))))
    #根据不同情况给出不同算法———————————————————————————————-----
    if s==2: 
        if n1==0: #若n1恰好是特殊的，则直接构造
            dd=np.zeros([N,0])
            wd_dd=0 
        elif n1 in [1,(ng-1)]: #若n1恰比行差设计多或少一列，直接给前n1列
            dd=XG[:,range(n1)] 
            wd_dd=WD_fast(dd) 
        else: #否则从XG中切割出来  
            (wd_dd, dd)=Cutting(N, n1, s, T, D0=XG)
    else:
        if n1==0: #若n1恰好是特殊的，则直接构造
            dd=np.zeros([N,0])
            wd_dd=0 
        elif n1 in [(ngg+1),(ng-1)]: #若n1恰比行差设计少一列，直接给前n1列
            dd=XG[:,range(n1)] 
            wd_dd=WD_fast(dd)
        elif n1 in [1,(ngg-1),ngg]:
            dd=XGG[:,range(n1)] 
            wd_dd=WD_fast(dd)
        else: #否则从XG中切割出来
            if n1 < ngg:
                (wd_dd, dd)=Cutting(N, n1, s, T, D0=XGG)
            else:
                (wd_dd, dd)=Cutting(N, n1, s, T, D0=XG)
    D=np.concatenate((D0, dd[random.sample(range(N), N),:]), axis=1) #并置D0和dd
    wd=float(Decimal(a) + Decimal(KK)*(Decimal(wd_dd) + (Decimal(4)/Decimal(3))**Decimal(n1) - (Decimal(3)/Decimal(2))**Decimal(n1)/Decimal(N)))
    return (wd, D)  

def COA_prime(N, n, s): #当水平数N为素数加1，构造COA(p(p-1),p)的转置再切一半视为XG，加一行后水平塌陷形成满足行差条件的设计
    #若p=2,N=3,所得的XG同GLP；所以此处只讨论p为奇素数,则N必为偶数，但s可以是奇数也可以是偶数
    p=N-1
    n0=int(p*(p-1)/2)
    k=int(n/n0) #需要k个XG并置
    n1=int(n-k*n0) #在并置基础上还需n1列, n1<n0
    Coa=np.zeros(shape=(p, p*(p-1))) #建立p行[p*(p-1)]列空矩阵
    for x in range(0,p):
        for i in range(1,p):
            for j in range(0,p):
                Coa[x][(i-1)*p+j] = mod(i*x+j, p) 
    COA=DeleteEC(Coa, p)[0] #把上面的Coa的等价列去除一半
    XG=np.concatenate((COA, p*np.ones([1, n0])), axis=0) #此处XG是(p+1)行[p*(p-1)/2]列的均匀LHD
    XG=Collapse(XG, s) #对XG水平塌陷
    D0=np.zeros([N,0])
    for i in range(0, k): #对XG随机行置换后再并置
        D0=np.concatenate((D0, XG[random.sample(range(N), N),:]), axis=1) #此时D0有k*n0=n-n1列   
    wd=float("inf")
    a = float(-(Decimal(4) / Decimal(3))**n + ((Decimal(3) / Decimal(2))**n)/Decimal(N))
    lamb_1=(N-s)*(k*n0)/(s*(N-1))
    lamb_2=N*(k*n0)/(s*(N-1))
    if int(s%2)==0: #若s是偶数
        Pr_even=1
        for i in range(1,int(s/2)):
            x=3/2-(i*(s-i))/(s**2)
            Pr_even=Pr_even*x 
        KK=float((Decimal(3)/Decimal(2))**Decimal(lamb_1) * (Decimal(5)/Decimal(4))**Decimal(lamb_2) * (Decimal(Pr_even)**(Decimal(2)*Decimal(lamb_2))))
    else: #若s是奇数
        Pr_odd=1
        for i in range(1,int((s+1)/2)):
            x=3/2-(i*(s-i))/(s**2)
            Pr_odd=Pr_odd*x       
        KK=float((Decimal(3)/Decimal(2))**Decimal(lamb_1) * (Decimal(Pr_odd)**(Decimal(2)*Decimal(lamb_2)))) 
    if n1==0: #若n1恰好是特殊的，则直接构造, wd_dd直接赋值
        dd=np.zeros([N,0])
        wd_dd=0 
    elif n1 in [1,(n0-1)]: #若n1恰比行差设计多或少一列，直接给前n1列
        dd=XG[:,range(n1)] 
        wd_dd=WD_fast(dd)
    D=np.concatenate((D0, dd[random.sample(range(N), N),:]), axis=1) #并置D0和dd
    wd=float(Decimal(a) + Decimal(KK)*(Decimal(wd_dd) + (Decimal(4)/Decimal(3))**Decimal(n1) - (Decimal(3)/Decimal(2))**Decimal(n1)/Decimal(N)))
    return (wd, D)


def PGLP(PowerG, Max_n0, N, n, s, T):  
    #要求:方幂GLP的生成向量已足够长,且所需设计行列比较大（若行列比N/n较小，效果不如Cutting函数）           
    X=FullFFD(N,1)
    G=PowerG[2][PowerG[1].index(Max_n0)] 
    XG=mod(np.array(mat(X)*mat(G)),N) #构造待切割的方幂GLP集
    (wd, D)=Cutting(N, n, s, T, D0 = XG)         
    return (wd, D)


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


def UniformDOE(N, n, s, T): 
    """    
Generating Uniform Designs of Experiments
    
Description:
    This function takes N, n, s and T to output a list(described below).

Parameters:
    N (int): The run size of the experiment (must be a multiple of s).
    n (int): The factor size of the experiment.
    s (int): The number of levels of the experiment.
    T (int): The maximum number of acceptable iterations.

Returns:
    array: A uniform or nearly uniform design under the wrap-around L2-discrepancy (WD). 
    float: The squared WD-value of the obtained design. 
    str: A hint of updating parameters for a better result.

Authors:
    Liangwei Qi (lwqi1996@126.com), Yongdao Zhou.

References:
    L. Qi, C. Ma and Y. Zhou. (2024). Systematic construction methods for uniform designs.

Examples:
    >>> UniformDOE(N=5, n=4, s=5, T=30)
        Hint: this squared WD-value achieves the lower bound!
        (0.13256182083950696,
         array([[0., 0., 1., 2.],
                [1., 2., 2., 4.],
                [2., 4., 3., 1.],
                [4., 3., 4., 3.],
                [3., 1., 0., 0.]]))
    >>> UniformDOE(N=9,n=5,s=3, T=30)
        Hint: Setting n as 4 or 8 will be a better choice!
        (0.33864407102575855,
         array([[0., 0., 0., 0., 0.],
                [2., 2., 1., 0., 2.],
                [0., 2., 2., 2., 0.],
                [1., 0., 1., 2., 1.],
                [2., 0., 2., 1., 2.],
                [1., 2., 0., 1., 1.],
                [0., 1., 1., 1., 0.],
                [2., 1., 0., 2., 1.],
                [1., 1., 2., 0., 2.]]))
    """
    if np.array([N,n,s,T]).dtype != int or N <= 1 or n <= 0 or s <=1 or T<=0:
        print("Error: A basic requirement is that N>1, n>0, s>1 and T>0, and they should all be positive integers!")
    elif N % s !=0: #若N不是s的倍数
        print("Error: N should be a multiple of s !")
    else: #若N是s的倍数
        if n==1:
            print("Hint: It is recommended that n>1!")
            (wd, D)=GeneratorMartix(N,n,s,T=1)
        elif Prime(s)==1: #若s是素数      
            if N>2: #N=2时s必为2，XG为一列，不需要用COA的方法
                n0=int((N-1)*(N-2)/2)
                k=int(n/n0) #需要k个COA的一半的并置
                n1=int(n-k*n0) #在并置基础上还需n1列,n1<n0
            if N>3 and Prime(N-1)==1 and n1 in [0, 1, n0-1]: #若能使用COA的转置
                #要求N>3是因为N=s=3时,COA程序得到的XG只有一列,不如GLP方法有意义
                (wd, D)=COA_prime(N, n, s)
                if n1==0: #对列数的说明和建议
                    print("Hint: this squared WD-value achieves the lower bound!")
                else:
                    print(f"Hint: Setting n as {int(n0*k)} or {int(n0*(k+1))} will be a better choice!")
            else: #若不能使用COA的转置
                if round(math.log(N,s), 10) % 1 == 0: #若N是s的幂
                    #用round防止math.log有小数的溢出,如math.log(125,5)
                    (wd, D)=GeneratorMartix(N,n,s,T)
                    n0=N-1
                    if s==2: #若s是偶素数 
                        if n % n0==0: #对列数的说明和建议
                            print("Hint: this squared WD-value achieves the lower bound!")
                        else:
                            print(f"Hint: Setting n as {int(n0*int(n/n0))} or {int(n0*(int(n/n0)+1))} will be a better choice!")
                    else: #若s是奇素数
                        if n % (n0/2)==0:
                            print("Hint: this squared WD-value achieves the lower bound!")
                        else:
                            print(f"Hint: Setting n as {int(n0/2*int(2*n/n0))} or {int(n0/2*(int(2*n/n0)+1))} will be a better choice!")
                else: #若N不是s的幂,从一个GLP或PGLP中切出来 
                    PowerG=PowerGen(N)
                    Max_n0=max(PowerG[1]) #方幂GLP所能容纳的最多列数
                    if n <= Max_n0 and N/n > 10: #用方幂GLP切割出来 
                        (wd, D)=PGLP(PowerG, Max_n0, N, n, s, T)      
                        if N==s:
                            p=N #找一个大于等于N的最小素数p,显然p是奇素数
                            while Prime(p)==0: 
                                p=p+1  
                            print(f"Hint: Setting N and s as {p} will be a better choice!")
                        else: #若N>s，则建议提升水平数
                            for i in list(range(s+1, int(N/2)+1))+[N]: #找N的比s大的因子i
                                if N % i ==0:
                                    break
                            print(f"Hint: Setting s as {i} will be a better choice!")                       
                    else: #若方幂GLP不可用        
                        n0=int((N-1)*(N-2)/2)
                        k=int(n/n0) #需要k个XG并置
                        n1=int(n-k*n0) #在并置基础上还需n1列,n1<n0
                        if Prime(N-1)==1 and n1 in [0, 1, n0-1]: #若能使用COA的转置
                            (wd, D)=COA_prime(N, n, s)
                            if n1==0: #对列数的说明和建议
                                print("Hint: this squared WD-value achieves the lower bound!")
                            else:
                                print(f"Hint: Setting n as {int(n0*k)} or {int(n0*(k+1))} will be a better choice!")
                        else:
                            (wd, D, p)=Cutting(N,n,s,T)
                            if N==s:
                                print(f"Hint: Setting N and s as {p} will be a better choice!")
                            else: #若N>s，则建议提升水平数
                                for i in list(range(s+1, int(N/2)+1))+[N]: #找N的比s大的因子i
                                    if N % i ==0:
                                        break
                                print(f"Hint: Setting s as {i} will be a better choice!")            
        else: #若s不是素数     
            PowerG=PowerGen(N)
            Max_n0=max(PowerG[1]) #方幂GLP所能容纳的最多列数
            if n <= Max_n0 and N/n > 10: #用方幂GLP切割出来 
                (wd, D)=PGLP(PowerG, Max_n0, N, n, s, T)      
                if N==s:
                    p=N #找一个大于等于N的最小素数p,显然p是奇素数
                    while Prime(p)==0: 
                        p=p+1  
                    print(f"Hint: Setting N and s as {p} will be a better choice!")
                else: #若N>s，则建议提升水平数
                    for i in list(range(s+1, int(N/2)+1))+[N]: #找N的比s大的因子i
                        if N % i ==0:
                            break
                    print(f"Hint: Setting s as {i} will be a better choice!")   
            else: #若方幂GLP不可用        
                n0=int((N-1)*(N-2)/2)
                k=int(n/n0) #需要k个XG并置
                n1=int(n-k*n0) #在并置基础上还需n1列,n1<n0
                if Prime(N-1)==1 and n1 in [0, 1, n0-1]: #若能使用COA的转置
                    (wd, D)=COA_prime(N, n, s)
                    if n1==0: #对列数的说明和建议
                        print("Hint: this squared WD-value achieves the lower bound!")
                    else:
                        print(f"Hint: Setting n as {int(n0*k)} or {int(n0*(k+1))} will be a better choice!")
                else:
                    (wd, D, p)=Cutting(N,n,s,T)
                    if N==s:
                        print(f"Hint: Setting N and s as {p} will be a better choice!")
                    else: #若N>s，则建议提升水平数
                        for i in list(range(s+1, int(N/2)+1))+[N]: #找N的比s大的因子i
                            if N % i ==0:
                                break
                        print(f"Hint: Setting s as {i} will be a better choice!")
        return (float(wd), D) 


# In[ ]:





# In[ ]:





# In[ ]:




