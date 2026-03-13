import os
# ==================== 环境配置 ====================
# 关闭嵌套并行，将底层库线程数设为1，避免在多进程任务中资源争抢导致效率下降
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from scipy.stats import qmc, wasserstein_distance
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# 忽略不必要的警告输出
warnings.filterwarnings("ignore")

# ==================== 通用工具函数 ====================

def mixture_kernel(u, v):
    """计算混合核函数 K^M，用于衡量样本偏差"""
    u = np.atleast_2d(u)
    v = np.atleast_2d(v)
    n1, s = u.shape
    n2, _ = v.shape
    KM = np.ones((n1, n2))
    
    for j in range(s):
        u_j = u[:, j].reshape(-1, 1)
        v_j = v[:, j].reshape(1, -1)
        term = 15.0/8.0 - 0.25*np.abs(u_j-0.5) - 0.25*np.abs(v_j-0.5) \
               - 0.75*np.abs(u_j-v_j) + 0.5*(u_j-v_j)**2
        KM *= term
    return KM

def squared_mixture_discrepancy(D):
    """计算混合偏差平方 (Squared Mixture Discrepancy)"""
    D = np.atleast_2d(D)
    n, s = D.shape
    term1 = (15.0/8.0)**s
    single_int = np.ones(n)
    for j in range(s):
        D_j = D[:, j].reshape(-1, 1)
        int_j = 15.0/8.0 - 0.25*np.abs(D_j-0.5) - 1.0/12.0
        single_int *= int_j.flatten()
    term2 = -2.0/n * np.sum(single_int)
    term3 = 1.0/(n**2) * np.sum(mixture_kernel(D, D))
    return term1 + term2 + term3

def truncate(samples):
    """将数据截断限制在 [0,1] 范围内"""
    return np.clip(samples, 0.0, 1.0)

def select_uniform_subsample_l1(full_X, n_uniform, l1_tree=None):
    """
    基于 L1 距离的均匀子抽样。
    """
    dim = full_X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sobol = qmc.Sobol(d=dim, scramble=True)
        uniform_points = sobol.random(n_uniform)
    
    # 构建或使用传入的 KDTree
    if l1_tree is None:
        tree = cKDTree(full_X)
    else:
        tree = l1_tree

    # 查询最近邻 (L1 范数)
    distances, indices = tree.query(uniform_points, k=1, p=1, workers=-1)
    indices = np.unique(indices)
    
    # 若去重后样本不足，随机补齐
    if len(indices) < n_uniform:
        remaining = n_uniform - len(indices)
        all_indices = np.arange(full_X.shape[0])
        mask = ~np.isin(all_indices, indices)
        if np.sum(mask) > 0:
            sup = np.random.choice(all_indices[mask], min(remaining, np.sum(mask)), replace=False)
            indices = np.concatenate([indices, sup])
    
    return indices[:n_uniform]

def iboss_subsampling(X, n):
    """
    IBOSS (Information-Based Optimal Sub-Sampling) 确定性采样。
    对每个特征排序并选取两端极值点。
    """
    N, p = X.shape
    indices = np.array([], dtype=int)
    r = int(np.ceil(n / p))
    if r % 2 != 0: r += 1 
    
    for j in range(p):
        if len(indices) >= n: break
        k = r // 2
        sorted_idx = np.argsort(X[:, j])
        # 选取该维度最小和最大的 k 个点
        idx_j = np.concatenate([sorted_idx[:k], sorted_idx[-k:]])
        indices = np.concatenate([indices, idx_j])
        
    indices = np.unique(indices)
    
    # 修正样本数量（截断或补齐）
    if len(indices) > n:
        indices = indices[:n]
    elif len(indices) < n:
        rem_count = n - len(indices)
        all_idx = np.arange(N)
        mask = np.isin(all_idx, indices, invert=True)
        fill_idx = all_idx[mask][:rem_count]
        indices = np.concatenate([indices, fill_idx])
        
    return indices

# ==================== DDS (Data-Driven Subsampling) ====================

def good_lattice_point_design(n, s, random_shift=True, random_state=None):
    """生成好格点 (Good Lattice Points) 设计，用于空间填充"""
    if n <= 0 or s <= 0: raise ValueError("n, s must be positive")
    rng = np.random.default_rng(random_state)
    p = n + 1
    valid_alphas = []
    # 寻找生成元 alpha
    for alpha in range(2, p):
        if np.gcd(alpha, p) != 1: continue
        powers = [(alpha**j) % p for j in range(1, s+1)]
        if len(set(powers)) == s: valid_alphas.append(alpha)
    
    if not valid_alphas:
        # 降级方案
        D = np.linspace(0, 1, n, endpoint=False).reshape(-1, 1)
        D = np.tile(D, (1, s))
    else:
        alpha = rng.choice(valid_alphas)
        gamma = [(alpha**j) % p for j in range(s)]
        indices = np.arange(1, n+1).reshape(-1, 1)
        D = (indices * gamma) % p
        D = D.astype(float)/n - 1.0/(2*n)
    
    if random_shift:
        D = (D + rng.random(s)) % 1
    return D

def dds_subsampling(X, n, var_threshold=0.85, random_shift=True, random_state=None):
    """
    DDS 采样算法。
    1. PCA 降维提取主要特征。
    2. 在降维空间生成好格点设计。
    3. 寻找最近邻样本。
    """
    X = np.atleast_2d(X)
    N, s_orig = X.shape
    rng = np.random.default_rng(random_state)
    
    # PCA 降维，保留累计方差贡献率 >= 85% 的维度
    if s_orig == 1:
        Z, s = X.copy(), 1
    else:
        pca = PCA()
        Z = pca.fit_transform(X)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        s = np.argmax(cum_var >= var_threshold) + 1
        s = max(s, 1)
        Z = Z[:, :s]
        
    # 生成设计点并映射回分位数空间
    D = good_lattice_point_design(n, s, random_shift=random_shift, random_state=random_state)
    Q_P = np.zeros_like(D)
    for j in range(s):
        Q_P[:, j] = np.quantile(np.sort(Z[:, j]), D[:, j])
        
    # KDTree 查询最近邻
    tree = cKDTree(Z)
    _, indices = tree.query(Q_P, k=1, workers=-1)
    indices = np.unique(indices.flatten())
    
    # 补齐样本
    if len(indices) < n:
        all_idx = np.arange(N)
        mask = ~np.isin(all_idx, indices)
        rem = n - len(indices)
        sup = rng.choice(all_idx[mask], rem, replace=False)
        indices = np.concatenate([indices, sup])
        
    return indices[:n]

# ==================== OSMAC (Optimal Subsampling) ====================

class OSMACLogistic:
    """逻辑回归的 OSMAC 两步采样法实现"""
    def __init__(self, criterion='mVc', fit_intercept=True):
        self.criterion = criterion # 优化准则: 'mVc' 或 'mMSE'
        self.fit_intercept = fit_intercept
        self.beta = None
    
    def _add_intercept(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def fit_weighted_logistic(self, X, y, weights=None):
        """带权重的逻辑回归 (使用 Newton-Raphson 迭代求解)"""
        from scipy.special import expit
        if weights is None: weights = np.ones(len(y))
        if self.fit_intercept: X = self._add_intercept(X)
        n_feat = X.shape[1]
        beta = np.zeros(n_feat)
        
        # 迭代求解 Beta
        for _ in range(20): 
            p = expit(X.dot(beta))
            grad = -X.T.dot(weights * (y - p))
            W = np.diag(weights * p * (1 - p))
            hess = X.T.dot(W).dot(X) + 1e-6*np.eye(n_feat) # 加正则项防奇异
            try: delta = np.linalg.solve(hess, grad)
            except: delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta -= delta
            if np.linalg.norm(delta) < 1e-6: break
        self.beta = beta
        return beta

    def predict(self, X):
        from scipy.special import expit
        if self.fit_intercept: X = self._add_intercept(X)
        return (expit(X.dot(self.beta)) >= 0.5).astype(int)

    def two_step_sampling(self, X, y, r0, r, random_state=None):
        """
        第一步：随机抽取 r0 个样本估算初步 Beta。
        第二步：计算最优抽样概率，抽取 r 个样本，合并后加权求解。
        """
        if random_state: np.random.seed(random_state)
        n = len(y)
        # Pilot sample
        idx0 = np.random.choice(n, r0, replace=False)
        w0 = np.ones(r0) * (n/r0)
        beta0 = self.fit_weighted_logistic(X[idx0], y[idx0], w0)
        
        if self.fit_intercept: X_d = self._add_intercept(X)
        else: X_d = X
        from scipy.special import expit
        p_hat = expit(X_d.dot(beta0))
        res = np.abs(y - p_hat)
        
        # 计算抽样得分与概率
        if self.criterion == 'mVc': score = res * np.linalg.norm(X_d, axis=1)
        else: score = res 
        prob = score / np.sum(score)
        prob = np.nan_to_num(prob, nan=1.0/n)
        prob /= prob.sum()
        
        # Second step sample
        idx1 = np.random.choice(n, r, p=prob, replace=False)
        X_c = np.vstack([X[idx0], X[idx1]])
        y_c = np.concatenate([y[idx0], y[idx1]])
        # 组合权重 (Inverse Probability Weighting)
        w_c = 1.0 / ((r0+r) * np.concatenate([np.ones(r0)/n, prob[idx1]]))
        self.fit_weighted_logistic(X_c, y_c, w_c)

class OSMACRegression:
    """线性回归的 OSMAC 两步采样法实现"""
    def __init__(self, criterion='mVc', fit_intercept=True):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.beta = None

    def _add_intercept(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit_weighted_ols(self, X, y, weights=None):
        """加权最小二乘法"""
        reg = LinearRegression(fit_intercept=self.fit_intercept)
        reg.fit(X, y, sample_weight=weights)
        if self.fit_intercept: self.beta = np.concatenate([[reg.intercept_], reg.coef_.flatten()])
        else: self.beta = reg.coef_.flatten()
        return self.beta

    def predict(self, X):
        X_d = self._add_intercept(X) if self.fit_intercept else X
        return X_d.dot(self.beta)

    def two_step_sampling(self, X, y, r0, r, random_state=None):
        """同逻辑回归的两步法流程"""
        if random_state: np.random.seed(random_state)
        n = len(y)
        idx0 = np.random.choice(n, r0, replace=False)
        w0 = np.ones(r0) * (n/r0)
        beta0 = self.fit_weighted_ols(X[idx0], y[idx0], weights=w0)
        
        X_d = self._add_intercept(X) if self.fit_intercept else X
        y_pred = X_d.dot(beta0)
        res = np.abs(y - y_pred)
        
        if self.criterion == 'mVc': score = res * np.linalg.norm(X_d, axis=1)
        else: score = res 
        prob = score / score.sum()
        prob = np.nan_to_num(prob, nan=1.0/n)
        prob /= prob.sum()
        
        idx1 = np.random.choice(n, r, p=prob, replace=False)
        X_c = np.vstack([X[idx0], X[idx1]])
        y_c = np.concatenate([y[idx0], y[idx1]])
        w_c = 1.0 / ((r0+r) * np.concatenate([np.ones(r0)/n, prob[idx1]]))
        self.fit_weighted_ols(X_c, y_c, weights=w_c)

# ==================== 数据生成模块 ====================

def get_2d_class_data(n_train=10000, n_test=2000):
    """生成 2D 分类数据 (混合高斯 + 均匀分布)"""
    mu = np.array([0.5, 0.5]); sigma2 = 0.3
    rng = np.random.default_rng(None)
    def gen(n):
        X = 0.8*truncate(rng.normal(mu, np.sqrt(sigma2), (n,2))) + 0.2*rng.uniform(0,1,(n,2))
        X = truncate(X)
        y = ((X[:,0]>0.5) & (X[:,1]>0.5)).astype(int)
        flip = rng.random(n) < 0.1 # 10% 噪声标签
        y[flip] = 1-y[flip]
        return X, y
    return *gen(n_train), *gen(n_test)

def get_10d_class_data(n_train=10000, n_test=2000):
    """生成 10D 分类数据 (两个高斯分布混合)"""
    dim=10
    mu1 = np.array([0.3]*5 + [0.7]*5)
    mu2 = np.array([0.7]*5 + [0.3]*5)
    sigma = np.zeros((dim,dim))
    # 构造协方差矩阵 AR(1) 结构
    for i in range(dim):
        for j in range(dim): sigma[i,j] = 0.2 if i==j else 0.1**np.abs(i-j)
    rng = np.random.default_rng(None)
    def gen(n):
        g1 = rng.multivariate_normal(mu1, sigma, n)
        g2 = rng.multivariate_normal(mu2, sigma, n)
        X = truncate(0.6*g1 + 0.4*g2)
        y = (np.sum(X>0.5, axis=1) >= 5).astype(int)
        flip = rng.random(n) < 0.1
        y[flip] = 1 - y[flip]
        return X, y
    return *gen(n_train), *gen(n_test)

def ackley_function(x):
    """Ackley 测试函数 (非凸优化常用)"""
    a, b, c = 20, 0.2, 2 * np.pi
    x_scaled = 65.536 * x - 32.768
    d = x.shape[1]
    sum_sq = np.sum(x_scaled**2, axis=1)
    sum_cos = np.sum(np.cos(c * x_scaled), axis=1)
    return -a * np.exp(-b * np.sqrt(sum_sq/d)) - np.exp(sum_cos/d) + a + np.e

def get_2d_reg_data(n_train=10000, n_test=2000):
    """生成 2D 回归数据 (基于 Ackley 函数)"""
    components = [(0.4, [0.0, 0.0], [1.0, 1.0]), (0.6, [0.2, 0.2], [0.8, 0.8])]
    def gen(n):
        weights = np.array([c[0] for c in components])
        weights /= weights.sum()
        indices = np.random.choice(len(components), size=n, p=weights)
        X = np.zeros((n, 2))
        for i, (_, low, high) in enumerate(components):
            mask = (indices == i)
            count = np.sum(mask)
            if count > 0:
                X[mask] = np.random.uniform(low, high, (count, 2))
        X[:, 1] = np.maximum(X[:, 1], 1e-6)
        y = ackley_function(X) + np.random.normal(0, 1.0, n)
        return X, y
    return *gen(n_train), *gen(n_test)

def get_10d_reg_data(n_train=10000, n_test=2000):
    """生成 10D 回归数据"""
    dim=10
    mu1 = np.array([1/3]*5 + [2/3]*5)
    mu2 = np.array([2/3]*5 + [1/3]*5)
    sigma = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim): sigma[i,j] = 0.2 if i==j else 0.1**np.abs(i-j)
    def gen(n):
        g1 = np.random.multivariate_normal(mu1, sigma, n)
        g2 = np.random.multivariate_normal(mu2, sigma, n)
        X = truncate(0.6*g1 + 0.4*g2)
        y = ackley_function(X) + np.random.normal(0,1,n)
        return X, y
    return *gen(n_train), *gen(n_test)

def get_5d_class_shift(n_train=10000, n_test=2000, noise=0.1):
    """生成带协变量偏移 (Shift) 的 5D 分类数据"""
    dim = 5
    # 训练集和测试集均值不同，制造分布差异
    mu1 = np.concatenate([np.full(3, 0.25), np.full(2, 0.75)])
    mu2 = 1.0 - mu1
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim): sigma[i, j] = 0.15 if i == j else (0.1) ** np.abs(i - j)
    
    def generate(n, mu, cov, noise_lvl):
        X = 0.7 * np.random.multivariate_normal(mu, cov, n) + 0.3 * np.random.uniform(0, 1, (n, dim))
        X = truncate(X)
        y = (np.sum(X >= 0.5, axis=1) >= 3).astype(int)
        flip_mask = np.random.rand(n) < noise_lvl
        y[flip_mask] = 1 - y[flip_mask]
        return X, y
    
    X_train, y_train = generate(n_train, mu1, sigma, noise)
    X_test, y_test = generate(n_test, mu2, sigma, noise)
    
    # 计算分布距离 (Wasserstein Distance)
    X_tr_w, _ = generate(100000, mu1, sigma, 0)
    X_te_w, _ = generate(100000, mu2, sigma, 0)
    w1 = sum(wasserstein_distance(X_tr_w[:, i], X_te_w[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1

def get_5d_reg_shift(n_train=10000, n_test=2000):
    """生成带协变量偏移的 5D 回归数据"""
    dim = 5
    mu1 = np.array([1/3, 1/3, 1/3, 2/3, 2/3])
    mu2 = np.array([2/3, 2/3, 2/3, 1/3, 1/3])  
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim): sigma[i, j] = 0.1 if i == j else (0.1) ** np.abs(i - j)
    def generate(n, mu, cov):
        g = np.random.multivariate_normal(mu, cov, n)
        u = np.random.uniform(0, 1, (n, dim))
        X = truncate(0.6 * g + 0.4 * u)
        y = ackley_function(X) + np.random.normal(0, 1, n)
        return X, y
    X_train, y_train = generate(n_train, mu1, sigma)
    X_test, y_test = generate(n_test, mu2, sigma)
    X_tr_w, _ = generate(100000, mu1, sigma)
    X_te_w, _ = generate(100000, mu2, sigma)
    w1 = sum(wasserstein_distance(X_tr_w[:, i], X_te_w[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1

# ==================== 实验 Runner ====================

def run_no_shift_exp(i, task_type, X_tr, y_tr, X_te, y_te, n_sub, rho_list):
    """
    运行无偏移场景下的对比实验。
    对比 Random, DDS, Uniform, OSMAC 及 SimSRT 方法。
    """
    np.random.seed(i)
    results = []
    # 回归任务增加多项式特征
    if task_type == 'Reg':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr_f = poly.fit_transform(X_tr)
        X_te_f = poly.transform(X_te)
    else:
        X_tr_f, X_te_f = X_tr, X_te
    
    # 建立 KDTree 用于均匀采样
    l1_tree = cKDTree(X_tr)
    
    def eval_model(idx):
        if task_type == 'Class':
            m = LogisticRegression(solver='liblinear', C=1.0)
            try:
                m.fit(X_tr_f[idx], y_tr[idx])
                return accuracy_score(y_te, m.predict(X_te_f))
            except: return 0.5
        else:
            m = LinearRegression()
            m.fit(X_tr_f[idx], y_tr[idx])
            return mean_squared_error(y_te, m.predict(X_te_f))

    # 1. Random Subsampling
    idx = np.random.choice(len(X_tr), n_sub, replace=False)
    results.append(('Random', np.nan, eval_model(idx)))
    
    # 2. DDS
    idx = dds_subsampling(X_tr, n_sub, random_state=i)
    results.append(('DDS', np.nan, eval_model(idx)))
    
    # 3. Uniform Subsampling
    idx = select_uniform_subsample_l1(X_tr, n_sub, l1_tree)
    results.append(('Uniform', np.nan, eval_model(idx)))
    
    # 4. OSMAC (mVc & mMSE)
    r0 = int(n_sub * 0.2)
    r1 = n_sub - r0
    for crit in ['mVc', 'mMSE']:
        if task_type == 'Class':
            osmac = OSMACLogistic(criterion=crit)
            osmac.two_step_sampling(X_tr_f, y_tr, r0, r1, random_state=i)
            score = accuracy_score(y_te, osmac.predict(X_te_f))
        else:
            osmac = OSMACRegression(criterion=crit)
            osmac.two_step_sampling(X_tr_f, y_tr, r0, r1, random_state=i)
            score = mean_squared_error(y_te, osmac.predict(X_te_f))
        results.append((f'OSMAC_{crit}', np.nan, score))
        
    # 5. SimSRT Subsampling (Uniform + Random)
    # rho 控制 Uniform 占比，rho越大，Uniform 占比越低 (n0/n1 = rho)
    for rho in rho_list:
        n0 = int(n_sub / (1 + rho)) # 随机部分
        n1 = n_sub - n0             # 均匀部分
        idx_uni = select_uniform_subsample_l1(X_tr, n1, l1_tree)
        rem = np.setdiff1d(np.arange(len(X_tr)), idx_uni)
        idx_rnd = np.random.choice(rem, n0, replace=False)
        combined = np.concatenate([idx_uni, idx_rnd])
        results.append(('SimSRT', rho, eval_model(combined)))
    return results

def run_shift_simsrt(i, task_type, X_tr, y_tr, X_te, y_te, n_sub, ratios):
    """
    运行偏移场景下的 SimSRT 测试。
    测试不同比例 (Random/Total) 下的模型表现。
    """
    np.random.seed(i + 1000)
    if task_type == 'Reg':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr_f = poly.fit_transform(X_tr)
        X_te_f = poly.transform(X_te)
    else:
        X_tr_f, X_te_f = X_tr, X_te
        
    l1_tree = cKDTree(X_tr)
    scores = []
    
    for r in ratios:
        n0 = int(n_sub * r) # 随机部分样本数
        n1 = n_sub - n0     # 均匀部分样本数
        idx_uni = select_uniform_subsample_l1(X_tr, n1, l1_tree)
        rem = np.setdiff1d(np.arange(len(X_tr)), idx_uni)
        
        if n0 > len(rem): idx_rnd = rem
        else: idx_rnd = np.random.choice(rem, n0, replace=False)
        combined = np.concatenate([idx_uni, idx_rnd])
        
        if task_type == 'Class':
            m = LogisticRegression(solver='liblinear', C=1.0)
            m.fit(X_tr_f[combined], y_tr[combined])
            score = accuracy_score(y_te, m.predict(X_te_f))
        else:
            m = LinearRegression()
            m.fit(X_tr_f[combined], y_tr[combined])
            score = mean_squared_error(y_te, m.predict(X_te_f))
        scores.append(score)
    return scores

# ==================== 主程序 ====================

def main():
    # --- 参数配置 ---
    N_REPEATS_NO_SHIFT = 1000  # 无偏移实验重复次数
    N_REPEATS_SHIFT = 1000     # 偏移实验重复次数
    RHO_LIST_NO_SHIFT = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0] # SimSRT方法的参数 rho 列表
    
    # 任务定义列表
    tasks_no_shift = [
        {'id': '2D_Class', 'type': 'Class', 'n': 500, 'gen': get_2d_class_data},
        {'id': '10D_Class', 'type': 'Class', 'n': 1500, 'gen': get_10d_class_data},
        {'id': '2D_Reg', 'type': 'Reg', 'n': 500, 'gen': get_2d_reg_data},
        {'id': '10D_Reg', 'type': 'Reg', 'n': 1500, 'gen': get_10d_reg_data},
    ]
    
    Ns_SHIFT = [300, 500, 1000, 1500]       # 偏移实验的样本量列表
    RATIOS = np.linspace(0.05, 0.95, 10)    # 随机采样占比 (n0/n)
    
    data_no_shift = []
    iboss_results = {}
    shift_results_list = []
    
    data_shift_class = {} 
    data_shift_reg = {}
    
    # --- Phase 1: 无偏移对照实验 ---
    print("=== Phase 1: 无偏移对照实验 (Rows 1 & 2) ===")
    for task in tasks_no_shift:
        print(f"Running {task['id']}...")
        X_tr, y_tr, X_te, y_te = task['gen']()
        
        # 计算全样本基准 (Full Benchmark)
        if task['type'] == 'Class':
            base = LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_te, y_te)
        else:
            deg = 2
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            Xt, Xte = poly.fit_transform(X_tr), poly.transform(X_te)
            base = mean_squared_error(y_te, LinearRegression().fit(Xt, y_tr).predict(Xte))
        
        # 计算 IBOSS 基准 (确定性算法，无需重复)
        idx_iboss = iboss_subsampling(X_tr, task['n'])
        if task['type'] == 'Class':
            m_iboss = LogisticRegression(solver='liblinear', C=1.0)
            m_iboss.fit(X_tr[idx_iboss], y_tr[idx_iboss])
            score_iboss = accuracy_score(y_te, m_iboss.predict(X_te))
        else:
            Xt_sub = Xt[idx_iboss]
            m_iboss = LinearRegression()
            m_iboss.fit(Xt_sub, y_tr[idx_iboss])
            score_iboss = mean_squared_error(y_te, m_iboss.predict(Xte))
        
        iboss_results[task['id']] = score_iboss

        # 并行运行主要对比实验
        out = Parallel(n_jobs=-1)(
            delayed(run_no_shift_exp)(i, task['type'], X_tr, y_tr, X_te, y_te, task['n'], RHO_LIST_NO_SHIFT)
            for i in range(N_REPEATS_NO_SHIFT)
        )
        for batch in out:
            for method, rho, score in batch:
                data_no_shift.append({
                    'Task': task['id'], 'Type': task['type'],
                    'Method': method, 'Rho': rho, 'Score': score, 'Base': base
                })

    # --- Phase 2: 分布偏移敏感性 (分类任务) ---
    print("=== Phase 2: 分布偏移敏感性 (Row 3: 5D Class) ===")
    X_tr_c, y_tr_c, X_te_c, y_te_c, w1_c = get_5d_class_shift()
    m_base = LogisticRegression(solver='liblinear').fit(X_tr_c, y_tr_c)
    base_class = accuracy_score(y_te_c, m_base.predict(X_te_c))
    
    for n_val in Ns_SHIFT:
        print(f"  Running n={n_val}...")
        out = Parallel(n_jobs=-1)(
            delayed(run_shift_simsrt)(i, 'Class', X_tr_c, y_tr_c, X_te_c, y_te_c, n_val, RATIOS)
            for i in range(N_REPEATS_SHIFT)
        )
        arr = np.array(out)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        data_shift_class[n_val] = {'r': RATIOS, 'mean': means, 'std': stds, 'base': base_class}
        
        for k in range(len(RATIOS)):
            shift_results_list.append({
                'TaskType': 'Class', 'N': n_val, 'Ratio': RATIOS[k], 
                'Mean': means[k], 'Std': stds[k], 'Base': base_class, 'W1': w1_c
            })

    # --- Phase 3: 分布偏移敏感性 (回归任务) ---
    print("=== Phase 3: 分布偏移敏感性 (Row 4: 5D Reg) ===")
    X_tr_r, y_tr_r, X_te_r, y_te_r, w1_r = get_5d_reg_shift()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xt_r = poly.fit_transform(X_tr_r)
    Xte_r = poly.transform(X_te_r)
    m_base = LinearRegression().fit(Xt_r, y_tr_r)
    base_reg = mean_squared_error(y_te_r, m_base.predict(Xte_r))
    
    for n_val in Ns_SHIFT:
        print(f"  Running n={n_val}...")
        out = Parallel(n_jobs=-1)(
            delayed(run_shift_simsrt)(i, 'Reg', X_tr_r, y_tr_r, X_te_r, y_te_r, n_val, RATIOS)
            for i in range(N_REPEATS_SHIFT)
        )
        arr = np.array(out)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        data_shift_reg[n_val] = {'r': RATIOS, 'mean': means, 'std': stds, 'base': base_reg}

        for k in range(len(RATIOS)):
            shift_results_list.append({
                'TaskType': 'Reg', 'N': n_val, 'Ratio': RATIOS[k], 
                'Mean': means[k], 'Std': stds[k], 'Base': base_reg, 'W1': w1_r
            })

    # ==================== 保存数据 ====================
    print("Saving experiment data to CSV...")
    df_no = pd.DataFrame(data_no_shift)
    df_no.to_csv('experiment_data_no_shift.csv', index=False)
    
    df_iboss = pd.DataFrame(list(iboss_results.items()), columns=['Task', 'Score'])
    df_iboss.to_csv('experiment_data_iboss.csv', index=False)
    
    df_shift = pd.DataFrame(shift_results_list)
    df_shift.to_csv('experiment_data_shift.csv', index=False)
    print("Data saved.")

    # ==================== 绘图 ====================
    print("Generating Plots...")
    
    # 4x4 网格布局
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=200)
    
    palette_main = {'Random': 'lightcoral', 'DDS': 'orange', 'Uniform': 'lightskyblue', 
                    'OSMAC_mMSE': '#90EE90', 'OSMAC_mVc': 'plum', 'SimSRT': '#DC2626'}
    method_order = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'OSMAC_mVc', 'SimSRT']
    method_colors = [palette_main[m] for m in method_order]
    
    layout_map = [(0,0), (0,2), (1,0), (1,2)]
    
    # --- ROW 1 & 2: 无偏移实验箱线图与 rho 敏感度 ---
    for i, task in enumerate(tasks_no_shift):
        tid = task['id']
        r, c = layout_map[i]
        df_sub = df_no[df_no['Task'] == tid].copy()
        is_class = (task['type'] == 'Class')
        
        # 寻找 SimSRT 方法的最佳 rho
        means = df_sub[df_sub['Method']=='SimSRT'].groupby('Rho')['Score'].mean()
        best_rho = means.idxmax() if is_class else means.idxmin()
        base_val = df_sub['Base'].iloc[0]
        
        # 1. 箱线图 (Method Comparison)
        ax_box = axes[r, c]
        mask = (df_sub['Method'].isin(method_order[:-1])) | \
               ((df_sub['Method'] == 'SimSRT') & (np.isclose(df_sub['Rho'], best_rho)))
        
        sns.boxplot(data=df_sub[mask], x='Method', y='Score', order=method_order, 
                    palette=palette_main, ax=ax_box, showfliers=False)
        
        # 添加基准线
        ax_box.axhline(base_val, color='crimson', ls='--', label='Full', zorder=10)
        ax_box.text(0.02, base_val, f'{base_val:.4f}', 
                    transform=ax_box.get_yaxis_transform(),
                    ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        
        if tid != '2D_Reg':
            iboss_val = iboss_results[tid]
            ax_box.axhline(iboss_val, color='black', ls='-.', linewidth=2, label='IBOSS', zorder=10)
            ax_box.text(0.02, iboss_val, f'{iboss_val:.4f}', 
                        transform=ax_box.get_yaxis_transform(),
                        ha='right', va='bottom', color='black', fontweight='bold', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        
        ax_box.set_title(f"{tid} (Without Shift)\nSimSRT $\\rho={best_rho}$")
        ax_box.set_xlabel('')
        ax_box.tick_params(axis='x', rotation=45)
        ax_box.set_ylabel('Accuracy' if is_class else 'MSE')
        ax_box.legend(loc='upper right')
        
        # 2. 敏感度图 (Rho Sensitivity)
        ax_sens = axes[r, c+1]
        sns.boxplot(data=df_sub[df_sub['Method']=='SimSRT'], x='Rho', y='Score', 
                    order=RHO_LIST_NO_SHIFT, palette=method_colors, ax=ax_sens, showfliers=False)
        
        ax_sens.axhline(base_val, color='crimson', ls='--', label='Full', zorder=10)
        ax_sens.text(0.02, base_val, f'{base_val:.4f}', 
                    transform=ax_sens.get_yaxis_transform(),
                    ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        
        ax_sens.set_title(f"{tid} - SimSRT Sensitivity")
        ax_sens.set_xlabel(r'$\rho$')
        ax_sens.set_ylabel('')
        ax_sens.legend(loc='upper right')

    def calc_ylim(mean_arr, std_arr, base_val, is_class=True):
        """动态计算 Y 轴范围"""
        upper = np.max(mean_arr + std_arr)
        lower = np.min(mean_arr - std_arr)
        upper = max(upper, base_val)
        lower = min(lower, base_val)
        diff = upper - lower
        if diff == 0: diff = 0.01
        margin = diff * 0.2
        y_min = lower - margin
        y_max = upper + margin
        if is_class:
             y_max = min(1.0, y_max)
             y_min = max(0.0, y_min)
        else:
             y_min = max(0.0, y_min)
        return y_min, y_max

    # --- ROW 3: 分类任务偏移实验绘图 ---
    for j, n_val in enumerate(Ns_SHIFT):
        ax = axes[2, j]
        res = data_shift_class[n_val]
        ax.plot(res['r'], res['mean'], '-o', color='steelblue', label='SimSRT')
        ax.fill_between(res['r'], res['mean']-res['std'], res['mean']+res['std'], 
                        alpha=0.2, color='steelblue')
        
        base_val = res['base']
        ax.axhline(base_val, color='crimson', ls='--', label='Full', zorder=10)
        ax.text(0.02, base_val, f'{base_val:.4f}', 
                transform=ax.get_yaxis_transform(),
                ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        
        y_min, y_max = calc_ylim(res['mean'], res['std'], res['base'], is_class=True)
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(f"5D Class Shift ($W_1={w1_c:.4f}$)\n$n={n_val}$")
        ax.set_xlabel('Random Ratio ($n_0/n$)')
        if j == 0: ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # --- ROW 4: 回归任务偏移实验绘图 ---
    for j, n_val in enumerate(Ns_SHIFT):
        ax = axes[3, j]
        res = data_shift_reg[n_val]
        
        ax.plot(res['r'], res['mean'], '-o', color='steelblue', label='SimSRT')
        ax.fill_between(res['r'], res['mean']-res['std'], res['mean']+res['std'], 
                        alpha=0.2, color='steelblue')
        
        base_val = res['base']
        ax.axhline(base_val, color='crimson', ls='--', label='Full', zorder=10)
        ax.text(0.02, base_val, f'{base_val:.4f}', 
                transform=ax.get_yaxis_transform(),
                ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        
        y_min, y_max = calc_ylim(res['mean'], res['std'], res['base'], is_class=False)
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(f"5D Reg Shift ($W_1={w1_r:.4f}$)\n$n={n_val}$")
        ax.set_xlabel('Random Ratio ($n_0/n$)')
        if j == 0: ax.set_ylabel('MSE')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig('Rho_without_shift_Visualization.png', dpi=300)
    plt.savefig('Rho_without_shift_Visualization.pdf', dpi=300)
    print("Done. Visualization saved.")

if __name__ == "__main__":
    main()