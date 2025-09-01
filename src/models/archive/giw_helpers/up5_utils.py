import numpy as np
import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_feature(net, train_loader, val_loader):
    net.eval()
    fe_tr_list = []
    fe_val_list = []
    index_val_list = []

    with torch.no_grad():
        # 提取训练集特征
        for features, _ in train_loader:
            features = features.float()
            if torch.cuda.is_available():
                features = features.cuda()
            _ = net(features)
            layer_output = activation['res_block2_output']
            fe_tr_list.append(layer_output.cpu().numpy())

        # 提取验证集特征
        idx = 0
        for features, _ in val_loader:
            features = features.float()
            if torch.cuda.is_available():
                features = features.cuda()
            _ = net(features)
            layer_output = activation['res_block2_output']
            fe_val_list.append(layer_output.cpu().numpy())
            batch_size = features.size(0)
            index_val_list.extend(range(idx, idx + batch_size))  #将当前批次中每个样本的全局索引添加到index_val_list中
            idx += batch_size

    fe_tr = np.concatenate(fe_tr_list, axis=0)
    fe_val = np.concatenate(fe_val_list, axis=0)

    return fe_tr, fe_val, index_val_list


def val_split(fe_tr, fe_val, index_val):
    clf = IsolationForest(random_state=42).fit(fe_tr) #表现更好
    # clf = OneClassSVM(gamma='auto').fit(fe_tr)

    # 获取验证集和训练集的得分
    w = clf.score_samples(fe_val)
    w_check = clf.score_samples(fe_tr)
    
    print("min Training Score:", np.min(w_check))
    print("min validation Score:", np.min(w))
    
    # 设置阈值，可以使用训练集得分的最小值
    threshold = np.min(w_check)
    
    # 根据阈值划分样本
    split_labels = []
    for score in w:
        if score >= threshold:
            split_labels.append(1)
        else:
            split_labels.append(0)
    
    # 计算被划分为正常类的样本比例
    alpha = np.count_nonzero(split_labels) / len(split_labels)
    val_dic = dict(zip(index_val, split_labels))
    
    return val_dic, alpha

def val_split_xy(X_train, y_train, X_val, y_val, index_val):
    # 将 X 和 y 进行拼接，形成联合分布
    fe_tr = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    fe_val = np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1)

    # 训练 Isolation Forest
    clf = IsolationForest(random_state=42).fit(fe_tr)

    # 获取验证集和训练集的得分
    w_val = clf.score_samples(fe_val)
    w_tr = clf.score_samples(fe_tr)

    print("min Training Score:", np.min(w_tr))
    print("min Validation Score:", np.min(w_val))

    # 设置阈值，可以使用训练集得分的最小值
    threshold = np.min(w_tr)

    # 根据阈值划分样本
    split_labels = (w_val >= threshold).astype(int)

    # 计算被划分为正常类的样本比例
    alpha = np.count_nonzero(split_labels) / len(split_labels)
    val_dic = dict(zip(index_val, split_labels))

    return val_dic, alpha