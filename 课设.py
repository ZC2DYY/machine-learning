# -*- coding: utf-8 -*-
#导入库
#基于stacking集成学习
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
#错误不报错
#读取数据
train_data = pd.read_csv('dataTrain.csv')
test_data = pd.read_csv('dataA.csv')
submission = pd.read_csv('submit_example_A.csv')
data_nolabel = pd.read_csv('dataNoLabel.csv')
print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape  = {test_data.shape}')
#自己特征
#自己构建一个f47列,也会影响最终结果
train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']
#ID类特征数值化
#字符流转成数字流，进行数值化
cat_columns = ['f3']
data = pd.concat([train_data, test_data])
for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])
#最后构造出训练集和测试集
num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]#除去id,label,f3三列，这三列不需要
feature_columns = num_columns + cat_columns#合并所有列
target = 'label'
train = train_data[feature_columns]#训练集输入x
label = train_data[target]#输出y
test = test_data[feature_columns]#测试集输入x
#常用的交叉验证模型框架
def model_train(model, model_name, kfold=5):
    #分为5份，4份训练，1份验证，轮流做小份验证集，训练集里重选训练集和验证集
    #预测值
    #数据存储
    oof_preds = np.zeros((train.shape[0]))#训练集
    test_preds = np.zeros(test.shape[0])#测试集
    skf = StratifiedKFold(n_splits=kfold)#调用函数，设置分几折
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):#k从1~5
        #训练集中定义新的训练集和测试集
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        model.fit(x_train,y_train)#进行训练
        y_pred = model.predict_proba(x_test)[:,1]#y的预测概率
        oof_preds[test_index] = y_pred.ravel()#进行数据摊开
        auc = roc_auc_score(y_test,y_pred)#设置数据
        print("- KFold = %d, val_auc = %.4f" % (k, auc))#输出不同k时的AUC得分
        test_fold_preds = model.predict_proba(test)[:, 1]#真正的test(无label）
        test_preds += test_fold_preds.ravel()#数据摊开
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold#得到一个真实平均答案，即submission.csv
#数据清洗
#设置60个数据来代表训练集中的60000个数据
gbc = GradientBoostingClassifier()
gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 60)
#发现AUC中后10个低于0.5,舍弃掉
train = train[:50000]
label = label[:50000]
#模型融合
gbc = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6,
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05,
    n_estimators=100,
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0
)
#设置参数
estimators = [
    ('gbc', gbc),
    ('hgbc', hgbc),
    ('xgbc', xgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
#两层堆积
clf = StackingClassifier(
    estimators=estimators,#第一层，调用5个分类器
    final_estimator=LogisticRegression()#第二层，逻辑回归算法
)
#特征筛选
#先将模型训练好，然后对验证集进行测试得到基础AUC，
# 之后循环遍历所有特征，在验证集上对单个特征进行mask后，得到mask后的AUC，
# 评估两个AUC的差值，差值越大，则说明特征重要性越高
#先将训练数据划分成训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)#随机种子
#然后用组合模型进行训练和验证(还没特征筛选的）
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)
#循环遍历特征，对验证集中的特征进行mask
ff = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))
#选取所有差值为负的特征，对比特征筛选后的特征提升
clf.fit(X_train[ff], y_train)
y_pred = clf.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)
#模型训练
train = train[ff]
test = test[ff]
clf_test_preds = model_train(clf, "StackingClassifier", 10)
submission['label'] = clf_test_preds
submission.to_csv('submission.csv', index=False)

