import os
import sys
import random
import numpy as np
import xgboost as xgb

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cvxopt import matrix, solvers
from scipy.spatial.distance import pdist

from sklearn.metrics import mean_squared_error, r2_score

############################
#   Utils: set_seed
############################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


############################
# (J) JointDistributionAdaptation: 直接处理连续标签的 JDA
############################
class JointDistributionAdaptation:
    """
    JDA: Joint Distribution Adaptation（连续标签版本）
    同时对齐源域和目标域的边缘分布与条件分布 P(Y|X)。
    本实现直接处理连续标签（例如标签在0到1之间），
    通过 RBF 标签核对齐条件分布，无需将连续标签离散化。
    
    参数:
        dim: 映射后子空间的维度
        kernel: 'linear' 或 'rbf'，用于特征的核计算
        gamma: 特征 RBF 核参数
        gamma_y: 标签 RBF 核参数（控制连续标签核带宽）
        nystrom_rank: Nyström 近似时选取的秩
    """
    def __init__(self, dim=50, kernel='rbf', gamma=1.0, gamma_y=1.0, nystrom_rank=300):
        self.dim = dim
        self.kernel = kernel
        self.gamma = gamma
        self.gamma_y = gamma_y
        self.nystrom_rank = nystrom_rank
        
        self.is_fitted = False
        self.X_ = None           # 合并后的数据
        self.eigvec_ = None      # 投影矩阵
        self.landmarks_ = None   # Nyström 近似时选取的支撑点

    def _kernel_func(self, X1, X2):
        """
        计算特征核矩阵，支持 'linear' 或 'rbf'
        """
        if self.kernel == 'linear':
            return X1.dot(X2.T)
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1**2, axis=1, keepdims=True)
            X2_sq = np.sum(X2**2, axis=1, keepdims=True)
            dist = X1_sq + X2_sq.T - 2 * X1.dot(X2.T)
            return np.exp(-self.gamma * dist)
        else:
            raise ValueError("Unsupported kernel.")

    def _label_kernel(self, Y1, Y2):
        """
        计算标签核矩阵，直接处理连续标签。
        采用 RBF 核：K_y(y1, y2) = exp(-gamma_y*(y1-y2)^2)
        输入 Y1, Y2 均为一维数组。
        """
        Y1 = Y1.reshape(-1, 1)
        Y2 = Y2.reshape(-1, 1)
        diff = Y1 - Y2.T
        return np.exp(-self.gamma_y * (diff ** 2))
    
    def _nystrom_approx(self, X):
        """
        使用 Nyström 近似计算映射:
          Zx = K(X, landmarks) * U * W^{-1/2}
        """
        K_xl = self._kernel_func(X, self.landmarks_)
        K_ll = self._kernel_func(self.landmarks_, self.landmarks_)
        vals, vecs = np.linalg.eigh(K_ll)
        vals[vals < 1e-12] = 1e-12
        W_inv_sqrt = np.diag((1.0 / vals) ** 0.5)
        Zx = K_xl.dot(vecs).dot(W_inv_sqrt)
        return Zx

    def _construct_M_continuous(self, Y_source, Y_target, ns, nt):
        """
        构造联合 MMD 矩阵 M（连续标签版本）
        其中 M = M_marginal + M_conditional，
        M_marginal 与原 TCA 类似；M_conditional 利用连续标签核对齐条件分布
        """
        n = ns + nt
        # 边缘分布项
        M_m = np.zeros((n, n))
        M_m[:ns, :ns] = 1.0 / (ns * ns)
        M_m[ns:, ns:] = 1.0 / (nt * nt)
        M_m[:ns, ns:] = -1.0 / (ns * nt)
        M_m[ns:, :ns] = -1.0 / (ns * nt)
        
        # 条件分布项：直接利用连续标签的核矩阵
        Y_source = np.asarray(Y_source).flatten()
        Y_target = np.asarray(Y_target).flatten()
        Kyy_ss = self._label_kernel(Y_source, Y_source)  # (ns, ns)
        Kyy_tt = self._label_kernel(Y_target, Y_target)  # (nt, nt)
        Kyy_st = self._label_kernel(Y_source, Y_target)  # (ns, nt)
        
        M_c = np.zeros((n, n))
        M_c[:ns, :ns] = Kyy_ss / (ns * ns)
        M_c[ns:, ns:] = Kyy_tt / (nt * nt)
        M_c[:ns, ns:] = -Kyy_st / (ns * nt)
        M_c[ns:, :ns] = -Kyy_st.T / (ns * nt)
        
        # 合并边缘和条件项
        M = M_m + M_c
        return M

    def fit_transform(self, Xs, Xt, Ys, Yt):
        """
        在源域 Xs 和目标域 Xt 及其连续标签 Ys, Yt 上进行 JDA 映射，
        返回映射后的 (Zs, Zt)
        """
        Xs = np.asarray(Xs)
        Xt = np.asarray(Xt)
        Ys = np.asarray(Ys)
        Yt = np.asarray(Yt)
        Xall = np.vstack([Xs, Xt])
        ns, nt = Xs.shape[0], Xt.shape[0]
        n = ns + nt
        self.X_ = Xall

        # Step 1: 使用 Nyström 近似
        r = min(self.nystrom_rank, n)
        self.landmarks_ = Xall[:r].copy()
        Zall = self._nystrom_approx(Xall)  # (n, r)

        # Step 2: 构造联合 MMD 矩阵（连续标签版本）
        M = self._construct_M_continuous(Ys, Yt, ns, nt)
        H = np.eye(n) - np.ones((n, n)) / n

        # Step 3: 求解广义特征值问题，使用 numpy 的 eigh 替换 ARPACK 方法
        left = Zall.T.dot(M).dot(Zall)
        right = Zall.T.dot(H).dot(Zall)
        left = 0.5 * (left + left.T)
        right = 0.5 * (right + right.T)
        reg = 1e-6 * np.eye(right.shape[0])
        A_mat = np.linalg.inv(right + reg).dot(left)
        A_mat = 0.5 * (A_mat + A_mat.T)
        eig_vals, eig_vecs = np.linalg.eigh(A_mat)
        idx = np.argsort(eig_vals)
        A = eig_vecs[:, idx[:self.dim]]  # (r, dim)
        self.eigvec_ = A
        Z = Zall.dot(A)              # (n, dim)

        self.is_fitted = True
        Zs_new = Z[:ns, :]
        Zt_new = Z[ns:, :]
        return Zs_new, Zt_new

    def transform(self, X):
        """
        对新数据 X 进行映射
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit_transform 方法.")
        X = np.asarray(X)
        Zx = self._nystrom_approx(X)  # (n_new, r)
        return Zx.dot(self.eigvec_)   # (n_new, dim)


############################
# (C) 改进 KMM: Nyström
############################
def nystrom_kmm(Xs, Xt, rank=300, B=10.0, eps=None, kernel='rbf', gamma=1.0):
    """
    改进版 KMM:
      - 用 Nyström 近似核, 减少对 (Xs, Xs) 全部 O(n_S^2) 运算
      - 最后构造 (ns, ns) 规模的矩阵用于 QP 求解
    """
    Xs = np.asarray(Xs)
    Xt = np.asarray(Xt)
    ns = len(Xs)
    nt = len(Xt)
    if eps is None:
        eps = 0.1 * ns

    # 1) 选取 landmarks
    r = min(rank, ns)
    landmarks = Xs[:r].copy()

    def kernel_func(A, B):
        if kernel == 'linear':
            return A.dot(B.T)
        elif kernel == 'rbf':
            A_sq = np.sum(A**2, axis=1, keepdims=True)
            B_sq = np.sum(B**2, axis=1, keepdims=True)
            dist = A_sq + B_sq.T - 2 * A.dot(B.T)
            return np.exp(-gamma * dist)
        else:
            raise ValueError("Unsupported kernel")
            
    # 2) 计算近似映射 Zs, Zt
    K_xl = kernel_func(Xs, landmarks)
    K_ll = kernel_func(landmarks, landmarks)
    vals, vecs = np.linalg.eigh(K_ll)
    vals[vals < 1e-12] = 1e-12
    W_inv_sqrt = np.diag((1.0 / vals)**0.5)
    Zs = K_xl.dot(vecs).dot(W_inv_sqrt)  # (ns, r)
    K_xt = kernel_func(Xt, landmarks)
    Zt = K_xt.dot(vecs).dot(W_inv_sqrt)  # (nt, r)

    # 3) 构造近似核矩阵
    Kss_approx = Zs.dot(Zs.T)  # (ns, ns)
    Kst_approx = Zs.dot(Zt.T)  # (ns, nt)
    Kss_approx += 1e-7 * np.eye(ns)
    kappa = -(float(ns) / float(nt)) * np.sum(Kst_approx, axis=1)

    # QP 构造
    P = Kss_approx
    q = kappa
    G_list = []
    h_list = []
    # sum_i beta_i <= ns + eps
    G_list.append(np.ones(ns))
    h_list.append(np.array([ns + eps]))
    # sum_i beta_i >= ns - eps
    G_list.append(-np.ones(ns))
    h_list.append(np.array([-(ns - eps)]))
    # 0 <= beta_i
    G_list.append(-np.eye(ns))
    h_list.append(np.zeros(ns))
    # beta_i <= B
    G_list.append(np.eye(ns))
    h_list.append(np.ones(ns) * B)
    G = np.vstack(G_list)
    h = np.concatenate(h_list)
    P_cvx = matrix(P, tc='d')
    q_cvx = matrix(q, tc='d')
    G_cvx = matrix(G, tc='d')
    h_cvx = matrix(h, tc='d')
    solvers.options['show_progress'] = False
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    beta = np.array(sol['x']).flatten()
    return beta


############################
# (D) XGBTrans 类: 基于 JDA+（可选 KMM 或直接使用原始特征）进行训练
############################
class XGBTrans:
    """
    简化示例:
      - 若无目标域 => 仅在源域上训练
      - 若有目标域 =>
          1) 如果 use_jda 为 True，则先对数据做 JDA 映射；
             如果 use_jda 为 False，则直接使用原始特征进入下一步；
          2) 如果 use_kmm 为 True，则利用 KMM 计算源域权重；
             如果 use_kmm 为 False，则直接使用均一权重；
          3) 合并源域和目标域数据后训练 XGBoost 模型。
    """
    def __init__(
        self,
        learning_rate=0.1,
        num_boost_round=200,
        seed=42,
        use_jda_nystrom=True,
        jda_nystrom_rank=1000,
        kmm_nystrom_rank=500,
        use_jda=True,    # 是否使用 JDA 映射；False 时直接使用原始特征
        use_kmm=False     # 是否使用 KMM 计算源域权重；False 时直接使用均一权重
    ):
        set_seed(seed)
        self.learning_rate = learning_rate
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.use_jda_nystrom = use_jda_nystrom
        self.jda_nystrom_rank = jda_nystrom_rank
        self.kmm_nystrom_rank = kmm_nystrom_rank
        self.use_jda = use_jda
        self.use_kmm = use_kmm

        self.bst = None
        self.jda = None   # 存放 JDA 映射对象
        self.beta = None
        self.pretrain_train_rmse = []
        self.pretrain_val_rmse = []
        self.pretrain_val_r2 = []
        self.finetune_train_rmse = []
        self.finetune_val_rmse = []
        self.finetune_val_r2 = []

    def _r2_metric(self, y_pred, dtrain):
        """
        自定义评估指标：计算 R²
        返回 (metric_name, metric_value)
        """
        y_true = dtrain.get_label()
        return ('r2', r2_score(y_true, y_pred))

    def train(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        # 如果没有验证集，则直接在源域训练
        if X_val is None or y_val is None:
            #print("    *[XGBTrans] No val => normal XGBoost on source domain only (pretrain).")
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': self.learning_rate,
                'seed': self.seed,
                'eval_metric': 'rmse'
            }
            evals_result = {}
            self.bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                evals_result=evals_result,
                feval=self._r2_metric,
                verbose_eval=False # LOG LEVEL
            )
            self.pretrain_train_rmse = evals_result['train']['rmse']
            self.pretrain_val_rmse = evals_result['eval']['rmse']
            self.pretrain_val_r2 = evals_result['eval']['r2']
            return

        # transfer learning 阶段
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        # 1) 使用 JDA 映射或直接使用原始特征
        if self.use_jda:
            #print("    *[XGBTrans] Use JDA => mapping source & val => reduce distribution gap.")
            self.jda = JointDistributionAdaptation(
                dim=6,
                kernel='rbf',
                gamma=1,
                gamma_y=1,  # 根据标签分布调整
                nystrom_rank=self.jda_nystrom_rank
            )
            Xs_map, Xv_map = self.jda.fit_transform(X_train, X_val, y_train, y_val)
            # 对测试集也做映射
            X_test_trans = self.jda.transform(X_test)
        else:
            #print("    *[XGBTrans] Skip JDA, use original features.")
            Xs_map, Xv_map = X_train, X_val
            X_test_trans = X_test
        
        # 2) 是否使用 KMM 计算权重
        if self.use_kmm:
            #print("    *[XGBTrans] Use Nyström KMM => get source weights.")
            self.beta = nystrom_kmm(
                Xs_map, Xv_map,
                rank=self.kmm_nystrom_rank,
                B=10.0,
                kernel='rbf',
                gamma=1.0
            )
            #print(f"KMM done => sum of beta={self.beta.sum():.2f}")
        else:
            self.beta = np.ones(len(Xs_map))
            #print("    *[XGBTrans] Skip KMM, use uniform weights for source.")

        # 3) 合并源域和目标域数据，并训练 XGBoost 模型
        X_combined = np.vstack([Xs_map, Xv_map])
        y_combined = np.concatenate([y_train, y_val])
        if self.use_kmm:
            w_combined = np.concatenate([self.beta, np.ones(len(y_val))])
        else:
            w_combined = np.ones(len(y_train) + len(y_val))
        
        dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
        dtest = xgb.DMatrix(X_test_trans, label=y_test)
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'eval_metric': 'rmse'
        }
        evals_result = {}
        self.bst = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            evals_result=evals_result,
            feval=self._r2_metric,
            verbose_eval=False # LOG LEVEL
        )
        self.finetune_train_rmse = evals_result['train']['rmse']
        self.finetune_val_rmse = evals_result['eval']['rmse']
        self.finetune_val_r2 = evals_result['eval']['r2']

    def predict(self, X):
        """
        预测: 如果使用了 JDA，则先 transform，否则直接预测
        """
        if self.use_jda and self.jda is not None and self.jda.is_fitted:
            X_ = self.jda.transform(X)
        else:
            X_ = X
        dtest = xgb.DMatrix(X_)
        return self.bst.predict(dtest)

    def get_epoch_metrics(self):
        metrics = {
            'loss_pretrain': self.pretrain_train_rmse,
            'test_loss_pretrain': self.pretrain_val_rmse,
            'r2_pretrain': self.pretrain_val_r2,
            'loss_finetune': self.finetune_train_rmse,
            'test_loss_finetune': self.finetune_val_rmse,
            'r2_finetune': self.finetune_val_r2
        }
        return metrics

    def save_model(self, filepath):
        if self.bst is None:
            print("No model to save.")
            return
        self.bst.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.bst = xgb.Booster()
        self.bst.load_model(filepath)
        print(f"Model loaded from {filepath}")
