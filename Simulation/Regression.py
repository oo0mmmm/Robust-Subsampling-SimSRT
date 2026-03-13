import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from joblib import Parallel, delayed
from scipy.stats import qmc, wasserstein_distance
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# ==================== 环境配置 ====================

# 限制线程数以防止并行计算时的资源争抢
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 无GUI环境（如服务器）请取消注释下行
# matplotlib.use('Agg') 

# ==================== DDS (Data Driven Subsampling) ====================

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

def good_lattice_point_design(n, s, random_shift=True, random_state=None):
    """生成好格子点 (GLP) 设计，用于低偏差采样"""
    if n <= 0 or s <= 0:
        raise ValueError("n和s必须为正整数")
    rng = np.random.default_rng(random_state)
    p = n + 1 
    valid_alphas = []
    # 寻找生成元
    for alpha in range(2, p):
        if np.gcd(alpha, p) != 1:
            continue
        powers = [(alpha**j) % p for j in range(1, s+1)]
        if len(set(powers)) == s:
            valid_alphas.append(alpha)
    
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
    """DDS子采样主函数：基于KD-Tree和GLP选择代表性样本"""
    X = np.atleast_2d(X)
    N, s_orig = X.shape
    if n <= 1 or n >= N:
        raise ValueError("n需满足1 < n < N")
    rng = np.random.default_rng(random_state)

    # 降维处理 (PCA)
    if precomputed_Z is not None and precomputed_s is not None:
        Z = precomputed_Z
        s = precomputed_s
    else:
        if s_orig == 1:
            Z = X.copy()
            s = 1
        else:
            pca = PCA()
            Z = pca.fit_transform(X)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            s = np.argmax(cum_var >= var_threshold) + 1 
            s = max(s, 1)
            Z = Z[:, :s]
    
    # 生成查询点并匹配最近邻
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
    
    # 补齐样本数量
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
    """IBOSS 确定性采样：基于特征排序选择极值点"""
    N, p = X.shape
    indices = np.array([], dtype=int)
    r = int(np.ceil(n / p))
    if r % 2 != 0: r += 1 
    
    for j in range(p):
        if len(indices) >= n: break
        k = r // 2
        sorted_idx = np.argsort(X[:, j])
        # 选择两端极值
        idx_j = np.concatenate([sorted_idx[:k], sorted_idx[-k:]])
        indices = np.concatenate([indices, idx_j])
        
    indices = np.unique(indices)
    
    # 修正样本量
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
    """OSMAC逻辑回归实现，支持mVc和mMSE准则"""
    def __init__(self, criterion='mVc', fit_intercept=True):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.beta = None
        self.M_X = None
        self.sampling_info = None 
    
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
        """牛顿-拉夫逊法求解加权逻辑回归"""
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
        """计算最优子采样概率"""
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
            raise ValueError("不支持的准则")
        
        probs_sum = np.sum(probs)
        if probs_sum == 0: probs = np.ones(n) / n
        else: probs = probs / probs_sum
        return probs
    
    def two_step_sampling(self, X, y, r0, r, initial_sampling='uniform', random_state=None):
        """两步采样流程：Pilot估计 -> 计算概率 -> 正式采样"""
        if random_state is not None: np.random.seed(random_state)
        n = len(y)
        if initial_sampling == 'uniform':
            prob0 = np.ones(n) / n
        elif initial_sampling == 'case-control':
            n0 = np.sum(y == 0); n1 = np.sum(y == 1)
            if n0 == 0 or n1 == 0: raise ValueError("类别不平衡错误")
            prob0 = np.where(y == 0, 1/(2*n0), 1/(2*n1))
        else:
            raise ValueError("初始化方法错误")
        
        # 第一步：Pilot采样
        indices0 = np.random.choice(n, size=r0, p=prob0, replace=False)
        X0 = X[indices0]; y0 = y[indices0]
        prob0_sampled = prob0[indices0]
        weights0 = 1 / (r0 * prob0_sampled) 
        beta0 = self.fit_weighted_logistic(X0, y0, weights=weights0)
        
        # 第二步：基于最优概率采样
        prob_optimal = self.calculate_subsampling_probs(X, y, beta0, self.criterion)
        indices1 = np.random.choice(n, size=r, p=prob_optimal, replace=False)
        X1 = X[indices1]; y1 = y[indices1]
        prob1_sampled = prob_optimal[indices1]
        
        # 合并与重加权
        X_combined = np.vstack([X0, X1])
        y_combined = np.concatenate([y0, y1])
        prob_combined = np.concatenate([prob0_sampled, prob1_sampled])
        weights_combined = 1 / ((r0 + r) * prob_combined) 
        final_beta = self.fit_weighted_logistic(X_combined, y_combined, weights=weights_combined)
        self.beta = final_beta
        self.sampling_info = {'indices0': indices0, 'indices1': indices1, 'beta0': beta0, 'prob_optimal': prob_optimal}
        return final_beta
    
    def predict_proba(self, X):
        from scipy.special import expit
        if self.beta is None: raise ValueError("模型未训练")
        if self.fit_intercept: X = self._add_intercept(X)
        z = X.dot(self.beta)
        return expit(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# ==================== 数据生成 ====================

SEED = 42
np.random.seed(SEED)
N_VALUES = [300, 500, 1000, 1500]
RHO_LIST = [0.5, 1.0, 2.0, 4.0]
RHO_FOR_BOXPLOT = [0.5, 1.0, 2.0, 4.0]
N_REPEATS = 50 

def truncate(samples):
    return np.clip(samples, 0.0, 1.0)

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    """Ackley函数：非凸优化测试函数"""
    x_scaled = 65.536 * x - 32.768
    d = x_scaled.shape[1]
    sum_sq = np.sum(x_scaled **2, axis=1)
    sum_cos = np.sum(np.cos(c * x_scaled), axis=1)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.e

def generate_mixture_uniform_2d(n_samples, components):
    """生成2D混合均匀分布数据"""
    X = np.zeros((n_samples, 2))
    weights = [comp[0] for comp in components]
    weights = np.array(weights) / np.sum(weights)
    for i in range(n_samples):
        component_idx = np.random.choice(len(components), p=weights)
        _, low, high = components[component_idx]
        sample = np.random.uniform(low, high, 2)
        sample[1] = max(sample[1], 1e-6)
        X[i] = sample
    return X

def get_2d_data(n_train=10000, n_test=2000, sigma_noise=1.0):
    """生成2D数据集（训练集和测试集分布不同）"""
    train_components = [(0.2, [0.0, 0.0], [1.0, 1.0]), (0.8, [0.1, 0.1], [0.7, 0.7])]
    test_components = [(0.2, [0.0, 0.0], [1.0, 1.0]), (0.8, [0.3, 0.3], [0.9, 0.9])]
    X_train = generate_mixture_uniform_2d(n_train, train_components)
    y_train = ackley_function(X_train) + np.random.normal(0, sigma_noise, n_train)
    X_test = generate_mixture_uniform_2d(n_test, test_components)
    y_test = ackley_function(X_test) + np.random.normal(0, sigma_noise, n_test)
    
    # 计算Wasserstein距离
    n_w1 = min(len(X_train), 20000)
    X_tr_w1 = generate_mixture_uniform_2d(n_w1, train_components)
    X_te_w1 = generate_mixture_uniform_2d(n_w1, test_components)
    w1_sum = sum(wasserstein_distance(X_tr_w1[:, i], X_te_w1[:, i]) for i in range(2))
    return X_train, y_train, X_test, y_test, w1_sum

def get_5d_data(n_train=10000, n_test=2000, noise=1.0):
    """生成5D数据集（高斯与均匀混合）"""
    dim = 5
    mu1 = np.array([1/3, 1/3, 1/3, 2/3, 2/3])
    mu2 = np.array([2/3, 2/3, 2/3, 1/3, 1/3])  
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j: sigma[i, j] = 0.1
            else: sigma[i, j] = (0.05) ** np.abs(i - j)
    def gen_weighted_sum(n, is_train):
        if is_train: mean_vec = mu1
        else: mean_vec = mu2
        X_gauss = np.random.multivariate_normal(mean=mean_vec, cov=sigma, size=n)
        X_unif = np.random.uniform(low=0.0, high=1.0, size=(n, dim))
        X = 0.6 * X_gauss + 0.4 * X_unif
        X = truncate(X)
        y = ackley_function(X) + np.random.normal(0, noise, n)
        return X, y
    X_train, y_train = gen_weighted_sum(n_train, True)
    X_test, y_test = gen_weighted_sum(n_test, False)
    
    n_w1 = min(len(X_train), 20000)
    X_tr_w1, _ = gen_weighted_sum(n_w1, True)
    X_te_w1, _ = gen_weighted_sum(n_w1, False)
    w1_sum = sum(wasserstein_distance(X_tr_w1[:, i], X_te_w1[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1_sum

def get_10d_data(n_train=10000, n_test=2000, noise=1.0):
    """生成10D数据集（高斯混合）"""
    dim = 10
    mu1 = np.array([1/5, 1/5, 1/5, 1/5, 1/5, 4/5, 4/5, 4/5, 4/5, 4/5]) 
    mu2 = np.array([4/5, 4/5, 4/5, 4/5, 4/5, 1/5, 1/5, 1/5, 1/5, 1/5])
    sigma = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j: sigma[i, j] = 0.05
            else: sigma[i, j] = (0.01) ** np.abs(i - j)
    def gen_biased(n, is_train):
        g1 = np.random.multivariate_normal(mean=mu1, cov=sigma, size=n)
        g2 = np.random.multivariate_normal(mean=mu2, cov=sigma, size=n)
        if is_train: X = 0.6 * g1 + 0.4 * g2
        else: X = 0.4 * g1 + 0.6 * g2
        X = truncate(X)
        y = ackley_function(X) + np.random.normal(0, noise, n)
        return X, y
    X_train, y_train = gen_biased(n_train, True)
    X_test, y_test = gen_biased(n_test, False)
    
    n_w1 = min(len(X_train), 20000)
    X_tr_w1, _ = gen_biased(n_w1, True)
    X_te_w1, _ = gen_biased(n_w1, False)
    w1_sum = sum(wasserstein_distance(X_tr_w1[:, i], X_te_w1[:, i]) for i in range(dim))
    return X_train, y_train, X_test, y_test, w1_sum

# ==================== 评估工具 ====================

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
    
    # 补足样本
    if len(indices) < n_uniform:
        remaining = n_uniform - len(indices)
        all_indices = np.arange(full_X.shape[0])
        mask = ~np.isin(all_indices, indices)
        unselected = all_indices[mask]
        if len(unselected) >= remaining:
            sup = np.random.choice(unselected, remaining, replace=False)
        else:
            sup = np.random.choice(unselected, len(unselected), replace=False)
        indices = np.concatenate([indices, sup])
    return indices[:n_uniform]

def fit_and_evaluate(indices, y_train, X_train_poly, X_test_poly, y_test, model_type='ols', best_alpha=None, random_state=None):
    """训练模型并计算测试集MSE"""
    X_sub = X_train_poly[indices]
    y_sub = y_train[indices]
    if model_type == 'ols': model = LinearRegression()
    elif model_type == 'ridge': model = Ridge(alpha=best_alpha, random_state=random_state)
    model.fit(X_sub, y_sub)
    y_pred = model.predict(X_test_poly)
    return mean_squared_error(y_test, y_pred)

def find_best_alpha_cv(X_train_poly, y_train, n_folds=5):
    """交叉验证选择最佳Ridge正则化系数"""
    alpha_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cv_scores = []
    for alpha in alpha_values:
        ridge = Ridge(alpha=alpha, random_state=SEED)
        scores = cross_val_score(ridge, X_train_poly, y_train, cv=n_folds, scoring='neg_mean_squared_error')
        cv_scores.append(np.mean(scores))
    best_idx = np.argmax(cv_scores)
    return alpha_values[best_idx]

# ============ 并行工作函数 ============

def run_trial(i, dim_name, X_train, y_train, X_test, y_test, n_val, rho_list, 
              best_alpha, X_train_poly, X_test_poly, 
              l1_tree=None, Z_dds=None, s_dds=None, tree_dds=None):
    """单次模拟：运行所有采样方法并返回评估指标"""
    
    current_seed = SEED + i + n_val 
    np.random.seed(current_seed)
    N = len(X_train)
    results = [] 
    def add_res(method, rho, penalty, mse):
        results.append({
            'Dimension': dim_name, 'n': n_val, 'Iter': i,
            'Method': method, 'Rho': rho, 'Penalty': penalty, 'MSE': mse
        })

    # 1. Random 采样
    idx_rand = np.random.choice(N, n_val, replace=False)
    add_res('Random', np.nan, 'OLS', fit_and_evaluate(idx_rand, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
    add_res('Random', np.nan, 'Ridge', fit_and_evaluate(idx_rand, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, current_seed))

    # 2. DDS 采样
    try:
        _, idx_dds = dds_subsampling(X_train, n_val, precomputed_Z=Z_dds, precomputed_s=s_dds, precomputed_tree=tree_dds, random_state=current_seed)
        add_res('DDS', np.nan, 'OLS', fit_and_evaluate(idx_dds, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
        add_res('DDS', np.nan, 'Ridge', fit_and_evaluate(idx_dds, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, current_seed))
    except:
        add_res('DDS', np.nan, 'OLS', np.nan); add_res('DDS', np.nan, 'Ridge', np.nan)

    # 3. Uniform 采样
    idx_uni = select_uniform_subsample_l1(X_train, n_val, l1_tree=l1_tree)
    add_res('Uniform', np.nan, 'OLS', fit_and_evaluate(idx_uni, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
    add_res('Uniform', np.nan, 'Ridge', fit_and_evaluate(idx_uni, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, current_seed))

    # 4. OSMAC 采样 (mVc & mMSE)
    def get_osmac(crit):
        try:
            osmac = OSMACLogistic(criterion=crit)
            r0 = max(1, int(n_val * 0.2))
            r = max(1, n_val - r0)
            osmac.two_step_sampling(X_train, y_train, r0, r, random_state=current_seed)
            return np.concatenate([osmac.sampling_info['indices0'], osmac.sampling_info['indices1']])
        except: return idx_rand
    
    idx_mvc = get_osmac('mVc')
    add_res('OSMAC_mVc', np.nan, 'OLS', fit_and_evaluate(idx_mvc, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
    add_res('OSMAC_mVc', np.nan, 'Ridge', fit_and_evaluate(idx_mvc, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, current_seed))

    idx_mmse = get_osmac('mMSE')
    add_res('OSMAC_mMSE', np.nan, 'OLS', fit_and_evaluate(idx_mmse, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
    add_res('OSMAC_mMSE', np.nan, 'Ridge', fit_and_evaluate(idx_mmse, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, current_seed))

    # 5. SimSRT 采样 (混合策略)
    for rho in rho_list:
        n0 = max(0, int(n_val / (1 + rho)))
        n1 = max(1, n_val - n0)
        n0 = n_val - n1 
        idx_rob_uni = select_uniform_subsample_l1(X_train, n1, l1_tree=l1_tree)
        idx_rob_rnd = np.random.choice(N, n0, replace=False)
        mask = ~np.isin(idx_rob_rnd, idx_rob_uni)
        idx_rob_rnd = idx_rob_rnd[mask]
        combined = np.concatenate([idx_rob_rnd, idx_rob_uni])
        if len(combined) < n_val:
            needed = n_val - len(combined)
            all_idx = np.arange(N)
            mask_all = ~np.isin(all_idx, combined)
            rem = all_idx[mask_all]
            if len(rem) >= needed:
                sup = np.random.choice(rem, needed, replace=False)
                combined = np.concatenate([combined, sup])
            else:
                combined = np.concatenate([combined, rem])
        combined = combined[:n_val]
        
        add_res('SimSRT', rho, 'OLS', fit_and_evaluate(combined, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
        add_res('SimSRT', rho, 'Ridge', fit_and_evaluate(combined, y_train, X_train_poly, X_test_poly, y_test, 'ols'))
        
    return results

# ==================== 可视化函数 ====================

def generate_indices_for_viz(X_train, n_val, viz_data):
    """为可视化生成各方法的采样索引"""
    indices_dict = {}
    y_temp = ackley_function(X_train)
    indices_dict['Random'] = np.random.choice(len(X_train), n_val, replace=False)
    _, idx_dds = dds_subsampling(X_train, n_val, precomputed_Z=viz_data['Z_dds'], precomputed_s=viz_data['s_dds'], precomputed_tree=viz_data['tree_dds'], random_state=42)
    indices_dict['DDS'] = idx_dds
    indices_dict['Uniform'] = select_uniform_subsample_l1(X_train, n_val, l1_tree=viz_data['l1_tree'])
    rho = 1.0
    n0 = int(n_val / (1 + rho)); n1 = n_val - n0
    idx_u = select_uniform_subsample_l1(X_train, n1, l1_tree=viz_data['l1_tree'])
    idx_r = np.random.choice(len(X_train), n0, replace=False)
    indices_dict['SimSRT'] = np.concatenate([idx_u, idx_r])[:n_val]
    
    # 增加 IBOSS 索引
    indices_dict['IBOSS'] = iboss_subsampling(X_train, n_val)
    
    try:
        osmac = OSMACLogistic(criterion='mVc')
        osmac.two_step_sampling(X_train, y_temp, int(n_val*0.2), int(n_val*0.8), random_state=42)
        indices_dict['OSMAC_mVc'] = np.concatenate([osmac.sampling_info['indices0'], osmac.sampling_info['indices1']])
    except: indices_dict['OSMAC_mVc'] = indices_dict['Random']
    try:
        osmac = OSMACLogistic(criterion='mMSE')
        osmac.two_step_sampling(X_train, y_temp, int(n_val*0.2), int(n_val*0.8), random_state=42)
        indices_dict['OSMAC_mMSE'] = np.concatenate([osmac.sampling_info['indices0'], osmac.sampling_info['indices1']])
    except: indices_dict['OSMAC_mMSE'] = indices_dict['Random']
    return indices_dict

def plot_combined_boxplots(df, n_values, rho_values, baselines, penalty_type='OLS'):
    """生成3x4组合箱线图 (2D/5D/10D)"""
    df_plot = df[df['Penalty'] == penalty_type].copy()
    
    methods = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'OSMAC_mVc', 'SimSRT']
    palette = {
        'Random': 'lightcoral', 'DDS': 'orange', 'Uniform': 'lightskyblue', 
        'OSMAC_mMSE': '#90EE90', 'OSMAC_mVc': 'plum', 'SimSRT': '#DC2626'
    }

    for n in n_values:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=300)
        fig.suptitle(f'MSE Comparison (n={n}, {penalty_type})', fontsize=20, y=0.99)
        
        df_n = df_plot[df_plot['n'] == n]
        dims = ['2D', '5D', '10D']
        
        for row, dim in enumerate(dims):
            full_mse = baselines.get((dim, n, 'Full', penalty_type), np.nan)
            iboss_mse = baselines.get((dim, n, 'IBOSS', penalty_type), np.nan)
            
            df_dim = df_n[df_n['Dimension'] == dim]

            for col, rho in enumerate(rho_values):
                ax = axes[row, col]
                
                mask_common = df_dim['Method'].isin(['Random', 'DDS', 'Uniform', 'OSMAC_mVc', 'OSMAC_mMSE'])
                mask_robust = (df_dim['Method'] == 'SimSRT') & (np.isclose(df_dim['Rho'], rho, atol=1e-5))
                
                df_sub = df_dim[mask_common | mask_robust].copy()
                df_sub['Method'] = pd.Categorical(df_sub['Method'], categories=methods, ordered=True)
                
                sns.boxplot(data=df_sub, x='Method', y='MSE', hue='Method', palette=palette, ax=ax, dodge=False, showfliers=False)
                
                # 绘制 Full Model 参考线
                if not np.isnan(full_mse):
                    ax.axhline(full_mse, color='crimson', linestyle='--', linewidth=2, label='Full', zorder=10)
                    ax.text(0.02, full_mse, f'{full_mse:.4f}', 
                            transform=ax.get_yaxis_transform(),
                            ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
                
                # 绘制 IBOSS 参考线
                if dim != '2D' and not np.isnan(iboss_mse):
                    ax.axhline(iboss_mse, color='black', ls='-.', linewidth=2, label='IBOSS', zorder=10)
                    ax.text(0.02, iboss_mse, f'{iboss_mse:.4f}', 
                            transform=ax.get_yaxis_transform(),
                            ha='left', va='bottom', color='black', fontweight='bold', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
                
                ax.set_title(f'{dim}, ρ={rho}')
                ax.set_xlabel('')
                if col == 0: ax.set_ylabel('Test MSE')
                else: ax.set_ylabel('')
                
                ax.tick_params(axis='x', rotation=45)
                if ax.get_legend() is not None: ax.get_legend().remove()
                ax.legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename1 = f'R_Combined_Boxplot_n{n}_{penalty_type}.pdf'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close()

def plot_layout_4rows_mixed(df, viz_data_2d, rho_values, baselines, penalty_type='OLS'):
    """生成混合布局图 (散点图 + 3行箱线图)"""
    df_plot = df[df['Penalty'] == penalty_type].copy()
    
    scatter_methods = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'IBOSS', 'SimSRT']
    boxplot_methods = ['Random', 'DDS', 'Uniform', 'OSMAC_mMSE', 'OSMAC_mVc', 'SimSRT']
    
    palette = {
        'Random': 'lightcoral', 'DDS': 'orange', 'Uniform': 'lightskyblue', 
        'OSMAC_mMSE': '#90EE90', 'OSMAC_mVc': 'plum', 'SimSRT': '#DC2626'
    }

    fig = plt.figure(figsize=(24, 20), dpi=300)
    gs = gridspec.GridSpec(4, 12, figure=fig)
    fig.suptitle(f'Polynomial Regression Subsampling({penalty_type})', fontsize=24, y=0.99)

    # === Part 1: 散点图 (Row 0) ===
    n_scatter = 300
    X_train = viz_data_2d['X_train']
    bg_idx = np.random.choice(len(X_train), int(len(X_train)*0.1), replace=False)
    X_bg = X_train[bg_idx]
    
    indices_map = generate_indices_for_viz(X_train, n_scatter, viz_data_2d)
    
    for i, method in enumerate(scatter_methods):
        ax = fig.add_subplot(gs[0, i*2 : i*2+2])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.scatter(X_bg[:,0], X_bg[:,1], c='lightblue', marker='o', s=30, alpha=0.7, label='Original')
        idx = indices_map[method]
        ax.scatter(X_train[idx, 0], X_train[idx, 1], c='plum', marker='^', s=50, alpha=0.8, label='Selected')
        ax.set_title(f'{method} (n={n_scatter})', fontsize=12)
        if i == 0: ax.set_ylabel('x2', fontsize=10)
        else: ax.set_yticks([]); ax.set_ylabel('')
        ax.set_xlabel('x1', fontsize=10)
        if i == 0: ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # === Part 2: 箱线图 (Rows 1-3) ===
    targets = [('2D', 500), ('5D', 1000), ('10D', 1500)]
    
    for row_idx, (dim, n_val) in enumerate(targets):
        grid_row = 1 + row_idx
        full_mse = baselines.get((dim, n_val, 'Full', penalty_type), np.nan)
        iboss_mse = baselines.get((dim, n_val, 'IBOSS', penalty_type), np.nan)
        df_target = df_plot[(df_plot['Dimension'] == dim) & (df_plot['n'] == n_val)]

        for col_idx, rho in enumerate(rho_values):
            ax = fig.add_subplot(gs[grid_row, col_idx*3 : col_idx*3+3])
            
            mask_common = df_target['Method'].isin(['Random', 'DDS', 'Uniform', 'OSMAC_mVc', 'OSMAC_mMSE'])
            mask_robust = (df_target['Method'] == 'SimSRT') & (np.isclose(df_target['Rho'], rho, atol=1e-5))
            df_sub = df_target[mask_common | mask_robust].copy()
            df_sub['Method'] = pd.Categorical(df_sub['Method'], categories=boxplot_methods, ordered=True)
            
            sns.boxplot(data=df_sub, x='Method', y='MSE', hue='Method', palette=palette, ax=ax, dodge=False, showfliers=False)
            
            if not np.isnan(full_mse):
                ax.axhline(full_mse, color='crimson', linestyle='--', linewidth=1.5, label='Full', zorder=10)
                ax.text(0.02, full_mse, f'{full_mse:.4f}', 
                        transform=ax.get_yaxis_transform(),
                        ha='left', va='bottom', color='crimson', fontweight='bold', fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
            
            if dim != '2D' and not np.isnan(iboss_mse):
                ax.axhline(iboss_mse, color='black', ls='-.', linewidth=2, label='IBOSS', zorder=10)
                ax.text(0.02, iboss_mse, f'{iboss_mse:.4f}', 
                        transform=ax.get_yaxis_transform(),
                        ha='left', va='bottom', color='black', fontweight='bold', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
            
            ax.set_title(f'{dim} (n={n_val}), ρ={rho}', fontsize=12)
            ax.set_xlabel('')
            if col_idx == 0: ax.set_ylabel('Test MSE', fontsize=12)
            else: ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=30)
            if ax.get_legend() is not None: ax.get_legend().remove()
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filename1 = f'R_Layout_4Rows_Mixed_{penalty_type}.pdf'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 主程序 ====================

def main():
    all_results_raw = [] 
    viz_data_2d = {} 
    
    # 存储基准线数据：Key: (Dimension, n, Method, Penalty), Value: MSE
    baselines = {}

    scenarios = [
        {'name': '2D', 'poly_deg': 3, 'func': get_2d_data},
        {'name': '5D', 'poly_deg': 2, 'func': get_5d_data},
        {'name': '10D', 'poly_deg': 2, 'func': get_10d_data}
    ]
    
    for sc in scenarios:
        dim_name = sc['name']
        print(f"\n{'='*40}\nProcessing Scenario: {dim_name}\n{'='*40}")
        
        X_train, y_train, X_test, y_test, w1 = sc['func']()
        print(f"Data generated. WD1 = {w1:.6f}")
        
        print("Precomputing KD-Tree and Polynomial Features...")
        l1_tree = cKDTree(X_train) 
        
        # DDS 预处理
        s_orig = X_train.shape[1]
        if s_orig == 1: Z_dds = X_train.copy(); s_dds = 1
        else:
            pca = PCA()
            Z_dds = pca.fit_transform(X_train)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            s_dds = np.argmax(cum_var >= 0.85) + 1
            s_dds = max(s_dds, 1)
            Z_dds = Z_dds[:, :s_dds]
        tree_dds = cKDTree(Z_dds) 
        
        if dim_name == '2D':
            viz_data_2d = {'X_train': X_train.copy(), 'l1_tree': l1_tree, 'Z_dds': Z_dds, 's_dds': s_dds, 'tree_dds': tree_dds}

        poly = PolynomialFeatures(degree=sc['poly_deg'], include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # 计算 Full Model 基准
        best_alpha = find_best_alpha_cv(X_train_poly, y_train)
        print(f"Best Ridge Alpha: {best_alpha}")

        full_ols = LinearRegression().fit(X_train_poly, y_train)
        mse_full_ols = mean_squared_error(y_test, full_ols.predict(X_test_poly))
        
        full_ridge = Ridge(alpha=best_alpha, random_state=SEED).fit(X_train_poly, y_train)
        mse_full_ridge = mean_squared_error(y_test, full_ridge.predict(X_test_poly))
        
        print(f"Full Model MSE - OLS: {mse_full_ols:.6f}, Ridge: {mse_full_ridge:.6f}")
        
        # 记录 Full Model 结果
        all_results_raw.append({'Dimension': dim_name, 'n': 0, 'Iter': 0, 'Method': 'Full_Model', 'Rho': np.nan, 'Penalty': 'OLS', 'MSE': mse_full_ols})
        all_results_raw.append({'Dimension': dim_name, 'n': 0, 'Iter': 0, 'Method': 'Full_Model', 'Rho': np.nan, 'Penalty': 'Ridge', 'MSE': mse_full_ridge})
        
        for n_val in N_VALUES:
            # 存入 Baseline
            baselines[(dim_name, n_val, 'Full', 'OLS')] = mse_full_ols
            baselines[(dim_name, n_val, 'Full', 'Ridge')] = mse_full_ridge
            
            # --- IBOSS 确定性采样计算 ---
            idx_iboss = iboss_subsampling(X_train, n_val)
            mse_iboss_ols = fit_and_evaluate(idx_iboss, y_train, X_train_poly, X_test_poly, y_test, 'ols')
            mse_iboss_ridge = fit_and_evaluate(idx_iboss, y_train, X_train_poly, X_test_poly, y_test, 'ridge', best_alpha, SEED)
            
            baselines[(dim_name, n_val, 'IBOSS', 'OLS')] = mse_iboss_ols
            baselines[(dim_name, n_val, 'IBOSS', 'Ridge')] = mse_iboss_ridge
            print(f"  n={n_val}: IBOSS MSE (OLS={mse_iboss_ols:.6f}, Ridge={mse_iboss_ridge:.6f})")

            all_results_raw.append({'Dimension': dim_name, 'n': n_val, 'Iter': 0, 'Method': 'IBOSS', 'Rho': np.nan, 'Penalty': 'OLS', 'MSE': mse_iboss_ols})
            all_results_raw.append({'Dimension': dim_name, 'n': n_val, 'Iter': 0, 'Method': 'IBOSS', 'Rho': np.nan, 'Penalty': 'Ridge', 'MSE': mse_iboss_ridge})

            # --- 运行随机采样实验 ---
            print(f"  Running stochastic methods n = {n_val} ({N_REPEATS} repeats)...")
            batch_results = Parallel(n_jobs=-1, verbose=0)(
                delayed(run_trial)(
                    i, dim_name, X_train, y_train, X_test, y_test, 
                    n_val, RHO_LIST, best_alpha, 
                    X_train_poly, X_test_poly,
                    l1_tree, Z_dds, s_dds, tree_dds
                )
                for i in range(N_REPEATS)
            )
            for batch in batch_results:
                all_results_raw.extend(batch)
                
    # 保存结果
    df_all = pd.DataFrame(all_results_raw)
    csv_filename = "simulation_results_raw.csv"
    try:
        df_all.to_csv(csv_filename, index=False)
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"simulation_results_raw_{timestamp}.csv"
        df_all.to_csv(new_filename, index=False)
    
    # 绘图 - OLS
    print("\nGenerating OLS Plots...")
    plot_combined_boxplots(df_all, N_VALUES, RHO_FOR_BOXPLOT, baselines, penalty_type='OLS')
    if viz_data_2d:
        plot_layout_4rows_mixed(df_all, viz_data_2d, RHO_FOR_BOXPLOT, baselines, penalty_type='OLS')
    
    # 绘图 - Ridge
    print("\nGenerating Ridge Plots...")
    plot_combined_boxplots(df_all, N_VALUES, RHO_FOR_BOXPLOT, baselines, penalty_type='Ridge')
    if viz_data_2d:
        plot_layout_4rows_mixed(df_all, viz_data_2d, RHO_FOR_BOXPLOT, baselines, penalty_type='Ridge')
    
    print("All tasks completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total Time: {time.time() - start_time:.2f} s")