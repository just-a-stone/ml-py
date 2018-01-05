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
# 标准化
scale_age_param = scaler.fit(X=X[['Age']])
X['scaled_age'] = scaler.transform(X=X[['Age']])

X = X.as_matrix()

y = X[:, 0]
X = X[:, 1:]

from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X=X, y=y)

print(clf.score(X=X, y=y))
