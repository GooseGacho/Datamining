# 特征工程
机器学习项目中有一句非常著名的话： 数据决定了模型的上限，算法只是逼近这个上限而已。
数据的质量对模型的精度和质量有决定性的影响。在实际中，原始数据有如下几个问题，必须经过处理后才能用于建模和训练使用：

 1. 原始数据中存在缺省值异常值等，在输入到模型进行训练之前，这些缺省值/异常值都需要处理
 2. 机器学习的模型只认识数值型的数据，而原始数据中的存在类别特征数据，例如表示性别的男女，表示地区的省份城市等。这些类别数据需要映射成数值才可以输入模型进行训练，即所谓的编码技术
 3. 原始数据中可能存在时间序列特征，大部分时候需要转化。例如在评估一辆二手车残值的时候，大部分时候将购买日期转化成已使用的时间更加合理
 4. 如果原始数据中特征太多，并且有一些字段可能和目标值关系不大，这时可以舍弃掉一些加快训练速度等。

提高数据质量是特征工程的终极目标。一个完整的特征工程分为数据预处理（Data Preprocessing）、特征构造（Feature Construction）、特征抽取（Feature Extraction）和特征选择（Feature Selection）几个步骤，每个步骤间没有明显的顺序之分，往往需要根据需求反复执行，甚至也没有严格区分的概念边界，例如特征构造可能会与数据预处理使用同样的数据变换技术。

缺省值/异常值处理、时间序列处理、特征编码和特征选择等常用方法


## 缺省值查看：
```py
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
train.isnull().sum()

>>>id                        0
loanAmnt                  0
term                      0
interestRate              0
installment               0
grade                     0
subGrade                  0
employmentTitle           1
employmentLength      46799
homeOwnership             0
annualIncome              0
verificationStatus        0
issueDate                 0
isDefault                 0
purpose                   0
postCode                  1
regionCode                0
dti                     239
delinquency_2years        0
ficoRangeLow              0
ficoRangeHigh             0
openAcc                   0
pubRec                    0
pubRecBankruptcies      405
revolBal                  0
revolUtil               531
totalAcc                  0
initialListStatus         0
applicationType           0
earliesCreditLine         0
title                     1
policyCode                0
n0                    40270
n1                    40270
n2                    40270
n3                    40270
n4                    33239
n5                    40270
n6                    40270
n7                    40270
n8                    40271
n9                    40270
n10                   33239
n11                   69752
n12                   40270
n13                   40270
n14                   40270
issueDateDT               0
dtype: int64
```
## 缺省值填充：
一般来说，缺失值是不可以直接删除的，因为：

 1. 真实场景中缺失值是正常现象，冒然删数据是很草率的行为，极端情况会把所有的数据删除；
 2. 被删除的数据可能正好是非常有用的数据，删除后会对模型质量有影响；
 
基于此，在机器学习项目中，都是对缺省值做填充处理。常见的填充方法有：
 
 
1. 0 值填充：即缺失值都填充为 0，这个方法简单，但是不推荐使用，尤其是那些有意义的数值特征如售价、房屋面积等。
2. 用缺失值上面或下面的值替换：这种填充的假设是相近的数据具有类似的特征。但这个假设不一定成立
3. 统计值填充：包括众数、平均值、中位数、四分位、八分位等。这个方法对于某些特征比较有效，比如用均值填充年纪的缺失值相对而言比较合理。这个也是比较通用的方案
4. 插值拟合：这个前提是假设我们有一列完整数据的特征，且该特征与有缺失值特征之间有一定关系，此时可以通过回归拟合得到缺失值。

### 按照平均数填充数值型特征
```py
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
test[numerical_fea] = test[numerical_fea].fillna(train[numerical_fea].median())
#按照众数填充类别型特征
train[category_fea] = train[category_fea].fillna(train[category_fea].mode())
test[category_fea] = test[category_fea].fillna(train[category_fea].mode())

train.isnull().sum()

id                        0
loanAmnt                  0
term                      0
interestRate              0
installment               0
grade                     0
subGrade                  0
employmentTitle           0
employmentLength      46799
homeOwnership             0
annualIncome              0
verificationStatus        0
issueDate                 0
isDefault                 0
purpose                   0
postCode                  0
regionCode                0
dti                       0
delinquency_2years        0
ficoRangeLow              0
ficoRangeHigh             0
openAcc                   0
pubRec                    0
pubRecBankruptcies        0
revolBal                  0
revolUtil                 0
totalAcc                  0
initialListStatus         0
applicationType           0
earliesCreditLine         0
title                     0
policyCode                0
n0                        0
n1                        0
n2                        0
n3                        0
n4                        0
n5                        0
n6                        0
n7                        0
n8                        0
n9                        0
n10                       0
n11                       0
n12                       0
n13                       0
n14                       0
issueDateDT               0
dtype: int64
```
可以看到数值型的特征已被处理

### 时间序列数据处理：
其中的issue为时间格式，需要转化
```py
转化成时间格式
for data in [train, test]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days
下面在对类别型的数据进行编码，即映射成数值型数据

category_fea
>>>['grade','subGrade','employmentLength','homeOwnership','verificationStatus','purpose']
```
下面直接调用sk-learn下面的preprocessing模块处理：
```py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
train['grade'] = le.fit_transform(train['grade'])
train['subGrade'] = le.fit_transform(train['subGrade'])
train['employmentLength'] = train['employmentLength'].apply(lambda x : str(x))
train['employmentLength'] = le.fit_transform(train['employmentLength'])
test['grade'] = le.fit_transform(test['grade'])
test['subGrade'] = le.fit_transform(test['subGrade'])
test['employmentLength'] = test['employmentLength'].apply(lambda x : str(x))
test['employmentLength'] = le.fit_transform(data['employmentLength'])
```
最后是对类别型特征进行转换，使其变为数值特征。包括两种情况：一种是对非数值特征数值化；另一种是对数值（这里的数值其实并没有 “数” 所代表的意义，只是个代码，所以要重新编码）编码。

具体有以下几种方法：

1. 序号编码：适用于类别间存在大小关系的特征。比如级别高中低，可以对应 321。
2. OneHot 编码：适用于不具有大小关系的特征。比如地名。
3. 二进制编码：先给每个类别赋予一个序号 ID，然后对 ID 进行二进制编码，最终得到和 OneHot 类似的 0-1 向量，但是维度更小。
 
经过上面操作后，在查看一下现在数据样式，可以发现所有特征已经转化成数值型的数据了，后期可以根据模型训练的效果，反复此步骤构建新的特征等。

另外，在实际应用中，随着对业务的深入理解我们可以通过现有的数据特征，创造新的特征，包括：

1. 衍生（升维）
2. 筛选（降维）


