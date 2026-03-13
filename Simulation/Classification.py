import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
from scipy.stats import qmc, wasserstein_distance
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# ==================== 环境设置 ====================

# 限制线程数以避免并行争抢资源
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 如果在服务器无头模式下运行，请取消注释
# matplotlib.use('Agg') 

# ==================== DDS (Data Driven Subsampling) ====================

def mixture_kernel(u, v):
    """计算混合偏差核函数 K^M"""
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
    """计算平方混合偏差"""
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

def good_lattice_point_design(n, s, random_shift=True, random_state=None):
    """生成好格子点 (GLP) 设计"""
    if n <= 0 or s <= 0: raise ValueError("n和s必须为正整数")
    rng = np.random.default_rng(random_state)
    p = n + 1 
    valid_alphas = []
    # 寻找有效的生成元
    for alpha in range(2, p):
        if np.gcd(alpha, p) != 1: continue
        powers = [(alpha**j) % p for j in range(1, s+1)]
        if len(set(powers)) == s: valid_alphas.append(alpha)
    
    if not valid_alphas:
        D = np.linspace(0, 1, n, endpoint=False).reshape(-1, 1)
        D = np.tile(D, (1, s))
    else:
        alpha = rng.choice(valid_alphas)
        gamma = [(alpha**j) % p for j in range(s)]
        indices = np.arange(1, n+1).reshape(-1, 1)
        D = (indices * gamma) % p
        D = D.astype(float)/n - 1.0/(2*n)
    
    if random_shift:
        epsilon = rng.random(s)
        D = (D + epsilon) % 1
    return D

def dds_subsampling(X, n, var_threshold=0.85, random_shift=True, random_state=None,
                    precomputed_Z=None, precomputed_s=None, precomputed_tree=None):
    """DDS子采样主逻辑"""
    X = np.atleast_2d(X)
    N, s_orig = X.shape
    if n <= 1 or n >= N: raise ValueError("n需满足1 < n < N")
    
    rng = np.random.default_rng(random_state)

    # 降维处理
    if precomputed_Z is not None and precomputed_s is not None:
        Z = precomputed_Z
        s = precomputed_s
    else:
        if s_orig == 1:
            Z = X.copy(); s = 1
        else:
            pca = PCA()
            Z = pca.fit_transform(X)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            s = np.argmax(cum_var >= var_threshold) + 1
            s = max(s, 1)
            Z = Z[:, :s]
    
    # 生成查询点并寻找最近邻
    D = good_lattice_point_design(n, s, random_shift=random_shift, random_state=random_state)
    Q_P = np.zeros_like(D)
    for j in range(s):
        z_sorted = np.sort(Z[:, j])
        Q_P[:, j] = np.quantile(z_sorted, D[:, j])

    if precomputed_tree is not None:
        tree = precomputed_tree
    else:
        tree = cKDTree(Z)
    
    distances, indices = tree.query(Q_P, k=1, workers=-1)
    subsample_indices = indices.flatten()
    subsample_indices = np.unique(subsample_indices)
    
    # 补足样本数量
    n_unique = len(subsample_indices)
    remaining_needed = n - n_unique
    
    if remaining_needed > 0:
        all_indices = np.arange(N)
        mask_unselected = ~np.isin(all_indices, subsample_indices)
        unselected_indices = all_indices[mask_unselected]
        if len(unselected_indices) >= remaining_needed:
            supplement_indices = rng.choice(unselected_indices, remaining_needed, replace=False)
            subsample_indices = np.concatenate([subsample_indices, supplement_indices])
        else:
            subsample_indices = np.concatenate([subsample_indices, unselected_indices])
            
    subsample_indices = subsample_indices[:n]
    subsample = X[subsample_indices, :]
    return subsample, subsample_indices

# ==================== IBOSS (Information-Based Optimal Sub-sampling) ====================

def iboss_subsampling(X, n):
    """IBOSS 确定性采样"""
    N, p = X.shape
    indices = np.array([], dtype=int)
    r = int(np.ceil(n / p))
    if r % 2 != 0: r += 1 
    
    for j in range(p):
        if len(indices) >= n: break
        k = r // 2
        sorted_idx = np.argsort(X[:, j])
        idx_j = np.concatenate([sorted_idx[:k], sorted_idx[-k:]])
        indices = np.concatenate([indices, idx_j])
        
    indices = np.unique(indices)
    
    if len(indices) > n:
        indices = indices[:n]
    elif len(indices) < n:
        rem_count = n - len(indices)
        all_idx = np.arange(N)
        mask = np.isin(all_idx, indices, invert=True)
        fill_idx = all_idx[mask][:rem_count]
        indices = np.concatenate([indices, fill_idx])
        
    return indices

# ==================== OSMAC (Optimal Subsampling Method) ====================

class OSMACLogistic:
    """OSMAC逻辑回归实现"""
    def __init__(self, criterion='mVc', fit_intercept=True):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.beta = None
        self.M_X = None
        self.sampling_info = None
        self.final_weights = None 
    
    def _add_intercept(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _weighted_logistic_loss(self, beta, X, y, weights):
        from scipy.special import expit
        z = X.dot(beta)
        p = expit(z)
        p = np.clip(p, 1e-10, 1-1e-10)
        log_likelihood = np.sum(weights * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        return -log_likelihood
    
    def _weighted_logistic_grad(self, beta, X, y, weights):
        from scipy.special import expit
        z = X.dot(beta)
        p = expit(z)
        return -X.T.dot(weights * (y - p))
    
    def _weighted_logistic_hess(self, beta, X, y, weights):
        from scipy.special import expit
        z = X.dot(beta)
        p = expit(z)
        W = np.diag(weights * p * (1 - p))
        return X.T.dot(W).dot(X)
    
    def fit_weighted_logistic(self, X, y, weights=None, max_iter=100, tol=1e-6):
        """使用牛顿-拉夫逊法求解加权逻辑回归"""
        if weights is None: weights = np.ones(len(y))
        if self.fit_intercept: X = self._add_intercept(X)
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)
        beta_curr = beta_init.copy()
        
        for i in range(max_iter):
            grad = self._weighted_logistic_grad(beta_curr, X, y, weights)
            hess = self._weighted_logistic_hess(beta_curr, X, y, weights)
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta_new = beta_curr - delta
            if np.linalg.norm(beta_new - beta_curr) < tol: break
            beta_curr = beta_new
        self.beta = beta_curr
        return self.beta
    
    def calculate_subsampling_probs(self, X, y, beta_pilot, criterion=None):
        """计算最优子采样概率 (mVc 或 mMSE)"""
        from scipy.special import expit
        if criterion is None: criterion = self.criterion
        if self.fit_intercept: X = self._add_intercept(X)
        n = X.shape[0]
        z = X.dot(beta_pilot)
        p = expit(z)
        p = np.clip(p, 1e-5, 1 - 1e-5)  
        abs_residuals = np.abs(y - p)
        
        if criterion == 'mVc':
            x_norms = np.linalg.norm(X, axis=1)
            probs = abs_residuals * x_norms
        elif criterion == 'mMSE':
            w = p * (1 - p)
            self.M_X = (X.T * w).dot(X) / n
            reg = 1e-6 * np.eye(self.M_X.shape[0])
            try:
                M_X_inv = np.linalg.inv(self.M_X + reg)
            except np.linalg.LinAlgError:
                M_X_inv = np.linalg.pinv(self.M_X + reg)
            M_X_norms = np.sqrt(np.sum((X.dot(M_X_inv) * X), axis=1))
            probs = abs_residuals * M_X_norms
        else:
            raise ValueError("Criterion error")
        
        probs_sum = np.sum(probs)
        if probs_sum == 0: probs = np.ones(n) / n
        else: probs = probs / probs_sum
        return probs
    
    def two_step_sampling(self, X, y, r0, r, initial_sampling='uniform', random_state=None):
        """执行两步采样：Pilot估计 -> 计算概率 -> 正式采样"""
        if random_state is not None: np.random.seed(random_state)
        n = len(y)
        if initial_sampling == 'uniform':
            prob0 = np.ones(n) / n
        elif initial_sampling == 'case-control':
            n0 = np.sum(y == 0); n1 = np.sum(y == 1)
            prob0 = np.where(y == 0, 1/(2*n0), 1/(2*n1))
        
        indices0 = np.random.choice(n, size=r0, p=prob0, replace=False)
        X0 = X[indices0]; y0 = y[indices0]
        prob0_sampled = prob0[indices0]
        weights0 = 1 / (r0 * prob0_sampled)
        beta0 = self.fit_weighted_logistic(X0, y0, weights=weights0)
        
        prob_optimal = self.calculate_subsampling_probs(X, y, beta0, self.criterion)
        indices1 = np.random.choice(n, size=r, p=prob_optimal, replace=False)
        X1 = X[indices1]; y1 = y[indices1]
        prob1_sampled = prob_optimal[indices1]
        
        X_combined = np.vstack([X0, X1])
        y_combined = np.concatenate([y0, y1])
        prob_combined = np.concatenate([prob0_sampled, prob1_sampled])
        weights_combined = 1 / ((r0 + r) * prob_combined)
        
        self.final_weights = weights_combined
        
        final_beta = self.fit_weighted_logistic(X_combined, y_combined, weights=weights_combined)
        
        self.beta = final_beta
        self.sampling_info = {'indices0': indices0, 'indices1': indices1, 'beta0': beta0, 'prob_optimal': prob_optimal}
        return final_beta

    def predict_proba(self, X):
        from scipy.special import expit
        if self.beta is None: raise ValueError("模型尚未训练")
        if self.fit_intercept: X = self._add_intercept(X)
        z = X.dot(self.beta)
        return expit(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# ==================== 实验参数与数据生成 ====================

SEED = 42
np.random.seed(SEED)

N_VALUES = [300, 500, 1000, 1500]
RHO_LIST = [0.5, 1.0, 2.0, 4.0]
RHO_FOR_BOXPLOT = [0.5, 1.0, 2.0, 4.0]
N_REPEATS = 1000

def truncate(samples):
    return np.clip(samples, 0.0, 1.0)

# --- 2D 数据生成 ---
def get_2d_data_classification(n_train=10000, n_test=2000, noise=0.1):
    dim = 2
    mu_tr = np.array([1/4, 1/4]); mu_te = np.array([3/4, 3/4]); sigma2 = 0.3

    def generate(n, mu, sigma_sq, noise_level):
        X_gauss = truncate(np.random.normal(loc=mu, scale=np.sqrt(sigma_sq), size=(n, dim)))
        X_unif = np.random.default_rng(42).uniform(0, 1, size=(n, dim))
        X = 0.75 * X_gauss + 0.25 * X_unif
        X = truncate(X)
        y = (np.sum(X, axis=1) > 0.5).astype(int)
        
        flip_mask = np.random.rand(n) < noise_level
        y[flip_mask] = 1 - y[flip_mask]
        return X, y

    X_train, y_train = generate(n_train, mu_tr, sigma2, noise)
    X_test, y_test = generate(n_test, mu_te, sigma2, noise)
    X_tr_w, _ = generate(100000, mu_tr, sigma2, 0)
    X_te_w, _ = generate(100000, mu_te, sigma2, 0)
    w1 = sum(wasserstein_distance(X_tr_w[:, i], X_te_w[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1

# --- 5D 数据生成 ---
def get_5d_data_classification(n_train=10000, n_test=2000, noise=0.1):
    dim = 5
    mu1 = np.concatenate([np.full(3, 0.25), np.full(2, 0.75)])
    mu2 = 1.0 - mu1
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim): sigma[i, j] = 0.25 if i == j else (0.15) ** np.abs(i - j)
        
    def generate(n, mu, cov, noise_level):
        X = 0.75 * np.random.multivariate_normal(mu, cov, n) + 0.25 * np.random.uniform(0, 1, (n, dim))
        X = truncate(X)
        y = (np.sum(X, axis=1) > 1.5).astype(int)
        
        flip_mask = np.random.rand(n) < noise_level
        y[flip_mask] = 1 - y[flip_mask]
        return X, y
        
    X_train, y_train = generate(n_train, mu1, sigma, noise)
    X_test, y_test = generate(n_test, mu2, sigma, noise)
    X_tr_w, _ = generate(100000, mu1, sigma, 0)
    X_te_w, _ = generate(100000, mu2, sigma, 0)
    w1 = sum(wasserstein_distance(X_tr_w[:, i], X_te_w[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1

# --- 10D 数据生成 ---
def get_10d_data_classification(n_train=10000, n_test=2000, noise=0.1):
    dim = 10
    mu1 = np.concatenate([np.full(5, 0.2), np.full(5, 0.8)])
    mu2 = 1.0 - mu1
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim): sigma[i, j] = 0.15 if i == j else (0.07) ** np.abs(i - j)
        
    def generate(n, mu_a, mu_b, cov, is_train, noise_level):
        X1 = np.random.multivariate_normal(mu_a, cov, n)
        X2 = np.random.multivariate_normal(mu_b, cov, n)
        X = (0.9 * X1 + 0.1 * X2) if is_train else (0.2 * X1 + 0.8 * X2)
        X = truncate(X)
        y = (np.sum(X, axis=1) > 4).astype(int)
        
        flip_mask = np.random.rand(n) < noise_level
        y[flip_mask] = 1 - y[flip_mask]
        return X, y
        
    X_train, y_train = generate(n_train, mu1, mu2, sigma, True, noise)
    X_test, y_test = generate(n_test, mu1, mu2, sigma, False, noise)
    X_tr_w, _ = generate(100000, mu1, mu2, sigma, True, 0)
    X_te_w, _ = generate(100000, mu1, mu2, sigma, False, 0)
    w1 = sum(wasserstein_distance(X_tr_w[:, i], X_te_w[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1

# ==================== 核心工具函数 ====================

def select_uniform_subsample_l1(full_X, n_uniform, l1_tree=None):
    """基于L1距离的均匀子采样"""
    dim = full_X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sobol = qmc.Sobol(d=dim, scramble=True)
        uniform_points = sobol.random(n_uniform)
    
    if l1_tree is None: tree = cKDTree(full_X)
    else: tree = l1_tree

    distances, indices = tree.query(uniform_points, k=1, p=1, workers=-1)
    indices = np.unique(indices)
    
    if len(indices) < n_uniform:
        remaining = n_uniform - len(indices)
        all_indices = np.arange(full_X.shape[0])
        mask = ~np.isin(all_indices, indices)
        sup = np.random.choice(all_indices[mask], remaining, replace=False)
        indices = np.concatenate([indices, sup])
    
    return indices[:n_uniform]

def fit_and_evaluate(indices, y_train, X_train, X_test, y_test, model_type='no_penalty', best_C=None, random_state=None, sample_weight=None):
    """训练模型并评估精度"""
    X_sub = X_train[indices]
    y_sub = y_train[indices]
    
    w_sub = None
    if sample_weight is not None:
        w_sub = sample_weight
    
    if model_type == 'no_penalty':
        model = LogisticRegression(penalty='l2', C=1e6, solver='liblinear', max_iter=100, tol=1e-4, random_state=random_state)
    elif model_type == 'l2':
        model = LogisticRegression(penalty='l2', C=best_C, solver='lbfgs', max_iter=300, tol=1e-6, random_state=random_state)
    
    try:
        model.fit(X_sub, y_sub, sample_weight=w_sub)
        return accuracy_score(y_test, model.predict(X_test))
    except: return 0.5

def find_best_C_cv(X_train, y_train, n_folds=5):
    """交叉验证寻找最佳C值"""
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cv_scores = []
    for C in C_values:
        lr = LogisticRegression(max_iter=300, C=C, penalty='l2', solver='lbfgs', tol=1e-6, random_state=42)
        scores = cross_val_score(lr, X_train, y_train, cv=n_folds, scoring='accuracy')
        cv_scores.append(np.mean(scores))
    best_idx = np.argmax(cv_scores)
    return C_values[best_idx], cv_scores[best_idx]

# ============ 并行实验任务 ============
def run_trial(i, dim_name, X_train, y_train, X_test, y_test, n_val, rho_list, 
              best_C, l1_tree=None, Z_dds=None, s_dds=None, tree_dds=None):
    
    current_seed = SEED + i + n_val 
    np.random.seed(current_seed)
    N = len(X_train)
    
    if l1_tree is None:
        l1_tree = cKDTree(X_train)
    
    results = [] 

    def add_res(method, rho, penalty, acc):
        results.append({'Dimension': dim_name, 'n': n_val, 'Iter': i, 'Method': method, 'Rho': rho, 'Penalty': penalty, 'ACC': acc})

    # --- 1. Random ---
    idx_rand = np.random.choice(N, n_val, replace=False)
    add_res('Random', np.nan, 'No Penalty', fit_and_evaluate(idx_rand, y_train, X_train, X_test, y_test, 'no_penalty', best_C, current_seed))
    add_res('Random', np.nan, 'L2 Penalty', fit_and_evaluate(idx_rand, y_train, X_train, X_test, y_test, 'l2', best_C, current_seed))

    # --- 2. DDS ---
    try:
        _, idx_dds = dds_subsampling(X_train, n_val, precomputed_Z=Z_dds, precomputed_s=s_dds, precomputed_tree=tree_dds, random_state=current_seed)
        add_res('DDS', np.nan, 'No Penalty', fit_and_evaluate(idx_dds, y_train, X_train, X_test, y_test, 'no_penalty', best_C, current_seed))
        add_res('DDS', np.nan, 'L2 Penalty', fit_and_evaluate(idx_dds, y_train, X_train, X_test, y_test, 'l2', best_C, current_seed))
    except:
        add_res('DDS', np.nan, 'No Penalty', 0.5)
        add_res('DDS', np.nan, 'L2 Penalty', 0.5)

    # --- 3. Uniform ---
    idx_uni = select_uniform_subsample_l1(X_train, n_val, l1_tree=l1_tree)
    add_res('Uniform', np.nan, 'No Penalty', fit_and_evaluate(idx_uni, y_train, X_train, X_test, y_test, 'no_penalty', best_C, current_seed))
    add_res('Uniform', np.nan, 'L2 Penalty', fit_and_evaluate(idx_uni, y_train, X_train, X_test, y_test, 'l2', best_C, current_seed))

    # --- 4. OSMAC ---
    def process_osmac(crit):
        try:
            osmac = OSMACLogistic(criterion=crit)
            r0 = max(1, int(n_val * 0.2))
            r = max(1, n_val - r0)
            osmac.two_step_sampling(X_train, y_train, r0, r, random_state=current_seed)
            
            # No Penalty: 使用OSMAC内部加权迭代的beta预测
            acc_no = accuracy_score(y_test, osmac.predict(X_test))
            
            # L2 Penalty: 使用sklearn重训，传入权重
            indices = np.concatenate([osmac.sampling_info['indices0'], osmac.sampling_info['indices1']])
            weights = osmac.final_weights 
            
            acc_l2 = fit_and_evaluate(indices, y_train, X_train, X_test, y_test, 'l2', best_C, current_seed, sample_weight=weights)
            return acc_no, acc_l2
        except: return 0.5, 0.5
            
    acc_mvc_no, acc_mvc_l2 = process_osmac('mVc')
    add_res('OSMAC_mVc', np.nan, 'No Penalty', acc_mvc_no)
    add_res('OSMAC_mVc', np.nan, 'L2 Penalty', acc_mvc_l2)

    acc_mmse_no, acc_mmse_l2 = process_osmac('mMSE')
    add_res('OSMAC_mMSE', np.nan, 'No Penalty', acc_mmse_no)
    add_res('OSMAC_mMSE', np.nan, 'L2 Penalty', acc_mmse_l2)

    # --- 5. SimSRT ---
    for rho in rho_list:
        n0 = max(0, int(n_val / (1 + rho))); n1 = max(1, n_val - n0)
        idx_simsrt_uni = select_uniform_subsample_l1(X_train, n1, l1_tree=l1_tree)
        idx_simsrt_rnd = np.random.choice(N, n0, replace=False)
        mask = ~np.isin(idx_simsrt_rnd, idx_simsrt_uni); idx_simsrt_rnd = idx_simsrt_rnd[mask]
        combined = np.concatenate([idx_simsrt_rnd, idx_simsrt_uni])
        if len(combined) < n_val:
            rem = np.setdiff1d(np.arange(N), combined)
            combined = np.concatenate([combined, np.random.choice(rem, n_val - len(combined), replace=False)])
        combined = combined[:n_val]
        
        add_res('SimSRT', rho, 'No Penalty', fit_and_evaluate(combined, y_train, X_train, X_test, y_test, 'no_penalty', best_C, current_seed))
        add_res('SimSRT', rho, 'L2 Penalty', fit_and_evaluate(combined, y_train, X_train, X_test, y_test, 'no_penalty', best_C, current_seed))
        
    return results

# ==================== 可视化函数 ====================

def generate_indices_for_viz(X_train, n_val, viz_data):
    """辅助函数：为可视化生成各方法的索引"""
    indices_dict = {}
    
    # Random
    indices_dict['Random'] = np.random.choice(len(X_train), n_val, replace=False)
    
    # DDS
    _, idx_dds = dds_subsampling(X_train, n_val, 
                                 precomputed_Z=viz_data['Z_dds'], 
                                 precomputed_s=viz_data['s_dds'], 
                                 precomputed_tree=viz_data['tree_dds'], 
                                 random_state=42)
    indices_dict['DDS'] = idx_dds
    
    # Uniform
    indices_dict['Uniform'] = select_uniform_subsample_l1(X_train, n_val, l1_tree=viz_data['l1_tree'])
    
    # SimSRT (Rho=1.0)
    rho = 1.0
    n0 = int(n_val / (1 + rho)); n1 = n_val - n0
    idx_u = select_uniform_subsample_l1(X_train, n1, l1_tree=viz_data['l1_tree'])
    idx_r = np.random.choice(len(X_train), n0, replace=False)
    indices_dict['SimSRT'] = np.concatenate([idx_u, idx_r])[:n_val]
    
    # IBOSS
    indices_dict['IBOSS'] = iboss_subsampling(X_train, n_val)
    
    # OSMAC_mMSE
    y_real = viz_data['y_train']
    try:
        osmac = OSMACLogistic(criterion='mMSE')
        osmac.two_step_sampling(X_train, y_real, int(n_val*0.2), int(n_val*0.8), random_state=42)
        indices_dict['OSMAC_mMSE'] = np.concatenate([osmac.sampling_info['indices0'], osmac.sampling_info['indices1']])
    except: indices_dict['OSMAC_mMSE'] = indices_dict['Random']
    
    return indices_dict

def plot_combined_boxplots(df, n_values, rho_values, baselines, penalty_type='No Penalty'):
    """生成组合箱线图"""
    df_plot = df[df['Penalty'] == penalty_type].copy()
    
    methods = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'OSMAC_mVc', 'SimSRT']

    palette = {
        'Random': 'lightcoral', 'DDS': 'orange', 'Uniform': 'lightskyblue', 
        'OSMAC_mMSE': '#90EE90', 'OSMAC_mVc': 'plum', 'SimSRT': '#DC2626'
    }

    for n in n_values:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=300)
        fig.suptitle(f'Accuracy Comparison (n={n}, {penalty_type})', fontsize=20, y=0.99)
        
        df_n = df_plot[df_plot['n'] == n]
        dims = ['2D', '5D', '10D']
        
        for row, dim in enumerate(dims):
            full_acc = baselines.get((dim, n, 'Full', penalty_type), np.nan)
            iboss_acc = baselines.get((dim, n, 'IBOSS', penalty_type), np.nan)
            
            df_dim = df_n[df_n['Dimension'] == dim]

            for col, rho in enumerate(rho_values):
                ax = axes[row, col]
                
                mask_common = df_dim['Method'].isin(methods)
                mask_robust = (df_dim['Method'] == 'SimSRT') & (np.isclose(df_dim['Rho'], rho, atol=1e-5))
                final_mask = (mask_common | mask_robust) & (df_dim['Method'].isin(methods))
                
                df_sub = df_dim[final_mask].copy()
                df_sub['Method'] = pd.Categorical(df_sub['Method'], categories=methods, ordered=True)
                
                sns.boxplot(data=df_sub, x='Method', y='ACC', hue='Method', palette=palette, ax=ax, dodge=False, showfliers=False)
                
                # 绘制基准线 (Full Model & IBOSS)
                if not np.isnan(full_acc):
                    ax.axhline(full_acc, color='crimson', linestyle='--', linewidth=2, label='Full', zorder=10)
                    ax.text(0.02, full_acc, f'{full_acc:.4f}', 
                            transform=ax.get_yaxis_transform(),
                            ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
                
                if not np.isnan(iboss_acc):
                    ax.axhline(iboss_acc, color='black', ls='-.', linewidth=2, label='IBOSS', zorder=10)
                    ax.text(0.02, iboss_acc, f'{iboss_acc:.4f}', 
                            transform=ax.get_yaxis_transform(),
                            ha='left', va='bottom', color='black', fontweight='bold', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
                
                ax.set_title(f'{dim}, ρ={rho}')
                ax.set_xlabel('')
                if col == 0: ax.set_ylabel('Test Accuracy')
                else: ax.set_ylabel('')
                
                ax.tick_params(axis='x', rotation=45)
                if ax.get_legend() is not None: ax.get_legend().remove()
                
                ax.legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'C_Combined_Boxplot_n{n}_{penalty_type}.pdf')
        plt.close()


def plot_layout_4rows_mixed(df, viz_data_2d, rho_values, baselines, penalty_type='No Penalty'):
    """生成混合布局图 (散点图 + 3行箱线图)"""
    df_plot = df[df['Penalty'] == penalty_type].copy()
    
    methods_box = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'OSMAC_mVc', 'SimSRT']
    methods_scatter = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'IBOSS', 'SimSRT']
    
    width_step = 2 

    palette = {
        'Random': 'lightcoral', 'DDS': 'orange', 'Uniform': 'lightskyblue', 
        'OSMAC_mMSE': '#90EE90', 'OSMAC_mVc': 'plum', 'SimSRT': '#DC2626'
    }

    fig = plt.figure(figsize=(24, 20), dpi=300)
    gs = gridspec.GridSpec(4, 12, figure=fig)
    fig.suptitle(f'Logistic Regression Subsampling({penalty_type})', fontsize=24, y=0.99)

    # 第一行: 散点图
    n_scatter = 300
    X_train = viz_data_2d['X_train']
    bg_idx = np.random.choice(len(X_train), int(len(X_train)*0.1), replace=False)
    X_bg = X_train[bg_idx]
    indices_map = generate_indices_for_viz(X_train, n_scatter, viz_data_2d)
    
    for i, method in enumerate(methods_scatter):
        ax = fig.add_subplot(gs[0, i*width_step : i*width_step + width_step])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.scatter(X_bg[:,0], X_bg[:,1], c='lightblue', marker='o', s=30, alpha=0.7, label='Original')
        
        if method in indices_map:
            idx = indices_map[method]
            ax.scatter(X_train[idx, 0], X_train[idx, 1], c='plum', marker='^', s=50, alpha=0.8, label='Selected')
        
        ax.set_title(f'{method} (n={n_scatter})', fontsize=12)
        if i == 0: ax.set_ylabel('x2', fontsize=10); 
        else: ax.set_yticks([]); ax.set_ylabel('')
        ax.set_xlabel('x1', fontsize=10)
        if i == 0: ax.legend(loc='upper right', fontsize=8)

    # 后三行: 箱线图
    targets = [('2D', 500), ('5D', 1000), ('10D', 1500)]
    for row_idx, (dim, n_val) in enumerate(targets):
        grid_row = 1 + row_idx
        full_acc = baselines.get((dim, n_val, 'Full', penalty_type), np.nan)
        iboss_acc = baselines.get((dim, n_val, 'IBOSS', penalty_type), np.nan)
        
        df_target = df_plot[(df_plot['Dimension'] == dim) & (df_plot['n'] == n_val)]

        for col_idx, rho in enumerate(rho_values):
            ax = fig.add_subplot(gs[grid_row, col_idx*3 : col_idx*3+3])
            
            mask_common = df_target['Method'].isin(methods_box)
            mask_robust = (df_target['Method'] == 'SimSRT') & (np.isclose(df_target['Rho'], rho, atol=1e-5))
            final_mask = (mask_common | mask_robust) & (df_target['Method'].isin(methods_box))

            df_sub = df_target[final_mask].copy()
            df_sub['Method'] = pd.Categorical(df_sub['Method'], categories=methods_box, ordered=True)
            
            sns.boxplot(data=df_sub, x='Method', y='ACC', hue='Method', palette=palette, ax=ax, dodge=False, showfliers=False)
            
            # 绘制基准线
            if not np.isnan(full_acc):
                ax.axhline(full_acc, color='crimson', linestyle='--', linewidth=1.5, label='Full', zorder=10)
                ax.text(0.02, full_acc, f'{full_acc:.4f}', 
                        transform=ax.get_yaxis_transform(),
                        ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
            
            if not np.isnan(iboss_acc):
                ax.axhline(iboss_acc, color='black', ls='-.', linewidth=2, label='IBOSS', zorder=10)
                ax.text(0.02, iboss_acc, f'{iboss_acc:.4f}', 
                        transform=ax.get_yaxis_transform(),
                        ha='left', va='bottom', color='black', fontweight='bold', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
            
            ax.set_title(f'{dim} (n={n_val}), ρ={rho}', fontsize=12)
            ax.set_xlabel('')
            if col_idx == 0: ax.set_ylabel('Test Accuracy', fontsize=12)
            else: ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=30)
            if ax.get_legend(): ax.get_legend().remove()
            
            ax.legend(loc='lower right')
            
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'C_Layout_4Rows_Mixed_{penalty_type}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 主程序 ====================

def main():
    all_results_raw = [] 
    viz_data_2d = {} 
    baselines = {}
    
    scenarios = [
        {'name': '2D', 'func': get_2d_data_classification},
        {'name': '5D', 'func': get_5d_data_classification},
        {'name': '10D', 'func': get_10d_data_classification}
    ]
    
    for sc in scenarios:
        dim_name = sc['name']
        print(f"\n{'='*40}\nProcessing: {dim_name} (Linear Features)\n{'='*40}")
        X_train, y_train, X_test, y_test, w1 = sc['func']()
        print(f"WD1 = {w1:.4f}")
        
        # 预计算 KD-Tree 和 PCA
        print("Precomputing KD-Tree & PCA...")
        l1_tree = cKDTree(X_train)
        
        s_orig = X_train.shape[1]
        if s_orig == 1:
            Z_dds = X_train.copy(); s_dds = 1
        else:
            pca = PCA()
            Z_dds = pca.fit_transform(X_train)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            s_dds = np.argmax(cum_var >= 0.85) + 1
            s_dds = max(s_dds, 1)
            Z_dds = Z_dds[:, :s_dds]
        tree_dds = cKDTree(Z_dds)
        
        # 保存 2D 数据用于后续可视化
        if dim_name == '2D':
            viz_data_2d['X_train'] = X_train.copy()
            viz_data_2d['y_train'] = y_train.copy()
            viz_data_2d['l1_tree'] = l1_tree
            viz_data_2d['Z_dds'] = Z_dds
            viz_data_2d['s_dds'] = s_dds
            viz_data_2d['tree_dds'] = tree_dds
        
        # 计算 Full Model 基准
        print("Finding best C for Full Model...")
        best_C, _ = find_best_C_cv(X_train, y_train)
        print(f"Best C (L2 Penalty): {best_C}")
        
        full_no = LogisticRegression(penalty='l2', C=1e6, solver='liblinear', max_iter=1000).fit(X_train, y_train)
        full_acc_no = accuracy_score(y_test, full_no.predict(X_test))
        print(f"Full Model (No Penalty) ACC: {full_acc_no:.6f}")
        
        full_l2 = LogisticRegression(penalty='l2', C=best_C, solver='lbfgs', max_iter=1000).fit(X_train, y_train)
        full_acc_l2 = accuracy_score(y_test, full_l2.predict(X_test))
        print(f"Full Model (L2 Penalty) ACC: {full_acc_l2:.6f}")
        
        all_results_raw.append({'Dimension': dim_name, 'n': 0, 'Iter': 0, 'Method': 'Full_Model', 'Rho': np.nan, 'Penalty': 'No Penalty', 'ACC': full_acc_no})
        all_results_raw.append({'Dimension': dim_name, 'n': 0, 'Iter': 0, 'Method': 'Full_Model', 'Rho': np.nan, 'Penalty': 'L2 Penalty', 'ACC': full_acc_l2})
        
        for n_val in N_VALUES:
            baselines[(dim_name, n_val, 'Full', 'No Penalty')] = full_acc_no
            baselines[(dim_name, n_val, 'Full', 'L2 Penalty')] = full_acc_l2
            
            # --- IBOSS 确定性采样 ---
            idx_iboss = iboss_subsampling(X_train, n_val)
            acc_iboss_no = fit_and_evaluate(idx_iboss, y_train, X_train, X_test, y_test, 'no_penalty', best_C, SEED)
            acc_iboss_l2 = fit_and_evaluate(idx_iboss, y_train, X_train, X_test, y_test, 'l2', best_C, SEED)
            
            baselines[(dim_name, n_val, 'IBOSS', 'No Penalty')] = acc_iboss_no
            baselines[(dim_name, n_val, 'IBOSS', 'L2 Penalty')] = acc_iboss_l2
            print(f"  n={n_val}: IBOSS ACC (NoPen={acc_iboss_no:.6f}, L2={acc_iboss_l2:.6f})")

            all_results_raw.append({'Dimension': dim_name, 'n': n_val, 'Iter': 0, 'Method': 'IBOSS', 'Rho': np.nan, 'Penalty': 'No Penalty', 'ACC': acc_iboss_no})
            all_results_raw.append({'Dimension': dim_name, 'n': n_val, 'Iter': 0, 'Method': 'IBOSS', 'Rho': np.nan, 'Penalty': 'L2 Penalty', 'ACC': acc_iboss_l2})

            # --- 运行随机采样实验 ---
            print(f"  Running stochastic methods n = {n_val} ({N_REPEATS} repeats)...")
            batch = Parallel(n_jobs=-1, verbose=0)(
                delayed(run_trial)(i, dim_name, X_train, y_train, X_test, y_test, n_val, RHO_LIST, best_C, l1_tree, Z_dds, s_dds, tree_dds)
                for i in range(N_REPEATS)
            )
            for b in batch: all_results_raw.extend(b)
                
    df_all = pd.DataFrame(all_results_raw)
    csv_filename = "simulation_results_corrected_final.csv"
    try:
        df_all.to_csv(csv_filename, index=False)
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"simulation_results_corrected_final_{timestamp}.csv"
        df_all.to_csv(new_filename, index=False)
    
    # 绘图 - No Penalty
    print("\nGenerating No Penalty Plots...")
    plot_combined_boxplots(df_all, N_VALUES, RHO_FOR_BOXPLOT, baselines, penalty_type='No Penalty')
    if viz_data_2d:
        plot_layout_4rows_mixed(df_all, viz_data_2d, RHO_FOR_BOXPLOT, baselines, penalty_type='No Penalty')
        
    # 绘图 - L2 Penalty
    print("\nGenerating L2 Penalty Plots...")
    plot_combined_boxplots(df_all, N_VALUES, RHO_FOR_BOXPLOT, baselines, penalty_type='L2 Penalty')
    if viz_data_2d:
        plot_layout_4rows_mixed(df_all, viz_data_2d, RHO_FOR_BOXPLOT, baselines, penalty_type='L2 Penalty')
        
    print("All tasks completed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"总耗时: {time.time() - start_time:.2f} 秒")