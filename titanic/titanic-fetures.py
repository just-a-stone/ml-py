import pandas as pd
from sklearn.preprocessing import StandardScaler

# read data
train_data = pd.read_csv(filepath_or_buffer='./data/train.csv')

# 数据补全
'''
1、Age缺失小部分      补全
2、Cabin缺失较多     丢弃     
3、Embarked缺失很少      补全
'''

# Age
# train_data['Age'].plot(kind='kde')
'''
1、选取有年龄数据
2、用randomForest做随机森林
3、剪枝
4、交叉验证
'''
# 假的，上面这些都没做

# 删除票号
X = train_data.drop(labels=['Cabin'], axis=1)
# 填充登陆港口，S港口登陆人数最多
X.loc[X['Embarked'].isnull(), 'Embarked'] = 'S'

# 分类变量get_dummies
dummies_sex = pd.get_dummies(data=X['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data=X['Pclass'], prefix='Pclass')
dummies_embarked = pd.get_dummies(data=X['Embarked'], prefix='Embarked')

# 数据组合
X = pd.concat(objs=[X, dummies_embarked, dummies_sex, dummies_pclass], axis=1)

# 票价 标准化
scaler = StandardScaler()
scale_fare_param = scaler.fit(X=X[['Fare']])
X['scaled_fare'] = scaler.transform(X=X[['Fare']])

# 年龄 标准化
# 以平均年龄填充缺失值
X.loc[X['Age'].isnull(), 'Age'] = X[X['Age'].notnull()]['Age'].mean()

# 构造新特征-isChild
X['isChild'] = pd.Series(1 if age <= 12 else 0 for age in X['Age'])

# age离散化
X['age'] = pd.cut(X.Age, bins=[0, 5, 15, 20, 35, 50, 60, 100],
                  labels=['age_0', 'age_5', 'age_15', 'age_20', 'age_35', 'age_50', 'age_60'])
dummies_age = pd.get_dummies(X['age'], prefix='age')
X = pd.concat([X, dummies_age], axis=1)

# 标准化
scale_age_param = scaler.fit(X=X[['Age']])
X['scaled_age'] = scaler.transform(X=X[['Age']])

# 构造新特征-family (未发现明显效果)
# X['family'] = X['SibSp'] + X['Parch']

# Name中提取称呼Mr,Mirss,Mrs,Dr... (没想到，称呼也能影响生存率)
X['name'] = X['Name'].str.extract('.+,(.+)\.', expand=False).str.strip()
dummies_name = pd.get_dummies(X['name'], prefix='name')
X = pd.concat([X, dummies_name], axis=1)

import numpy as np

# Ticket中提取仓位信息（前面带字母的，是不是高端些）
X['Ticked_letter'] = X['Ticket'].str.split().str[0]
X['Ticked_letter'] = X['Ticked_letter'].apply(lambda x: np.nan if x.isnumeric() else x)
X = pd.get_dummies(X, prefix='ticket', columns=['Ticked_letter'], drop_first=True)

# 变量筛选
# regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild'
# regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild|SibSp|Parch'
# regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild|SibSp|Parch|age_.*'
# regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild|SibSp|Parch|age_.*|family'
# regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild|SibSp|Parch|age_.*|name_.*'
regex = 'Survived|Pclass_.*|Sex_.*|scaled_.*|Embarked_.*|isChild|SibSp|Parch|age_.*|name_.*|ticket_.*'
X_final = pd.DataFrame(X).filter(regex=regex)

X = X_final.as_matrix()

y = X[:, 0]
X = X[:, 1:]

from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X=X, y=y)

print(clf.score(X=X, y=y))
print(pd.DataFrame({'feature': list(X_final.columns[1:]), 'weight': list(clf.coef_.T)}))

# 尝试交叉验证
# clf = linear_model.LogisticRegressionCV()
# clf.fit(X, y)
#
# print(clf.score(X, y))
