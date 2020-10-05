# 探索性数据分析，即EDA，Exploratory Data Analysis。
一般来说，数据描述性统计分析是一个数据挖掘/机器学习项目的第一步，有时候也被称为探索性数据分析。但两者有一些细微的区别：

数据描述统计强调方法，即如何从数据中获取信息。比如用平均数／中位数／众数／方差等描述城市的人员收入水平。
数据探索强调过程，即通过对数据进行研究发现其规律，对研究的对象有更加深入的认识。例如通过对不同人员的工作年限行业教育水平和薪资的关系，找到影响收入的因素等

## 1. 缺省值查看

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./train.csv')
test = pd.read_csv('./testA.csv')
```
## 查看缺省值
```python
d = (train.isnull().sum()/len(train)).to_dict()
print(d)
print('*'*30)
print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')
```
```
{'id': 0.0, 'loanAmnt': 0.0, 'term': 0.0, 'interestRate': 0.0, 'installment': 0.0, 'grade': 0.0, 'subGrade': 0.0, 'employmentTitle': 1.25e-06, 'employmentLength': 0.05849875, 'homeOwnership': 0.0, 'annualIncome': 0.0, 'verificationStatus': 0.0, 'issueDate': 0.0, 'isDefault': 0.0, 'purpose': 0.0, 'postCode': 1.25e-06, 'regionCode': 0.0, 'dti': 0.00029875, 'delinquency_2years': 0.0, 'ficoRangeLow': 0.0, 'ficoRangeHigh': 0.0, 'openAcc': 0.0, 'pubRec': 0.0, 'pubRecBankruptcies': 0.00050625, 'revolBal': 0.0, 'revolUtil': 0.00066375, 'totalAcc': 0.0, 'initialListStatus': 0.0, 'applicationType': 0.0, 'earliesCreditLine': 0.0, 'title': 1.25e-06, 'policyCode': 0.0, 'n0': 0.0503375, 'n1': 0.0503375, 'n2': 0.0503375, 'n3': 0.0503375, 'n4': 0.04154875, 'n5': 0.0503375, 'n6': 0.0503375, 'n7': 0.0503375, 'n8': 0.05033875, 'n9': 0.0503375, 'n10': 0.04154875, 'n11': 0.08719, 'n12': 0.0503375, 'n13': 0.0503375, 'n14': 0.0503375}
```




## 可视化
```python
(train.isnull().sum()/len(train)).plot.bar(figsize = (20,6))
```
![1.png](https://img-blog.csdnimg.cn/20201005212815575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)

可以看到所有的特征缺失值都在10%以内，全部保留。

总结
 47列数据中有22列都缺少少量数据，这在现实世界中很正常。‘policyCode’具有一个唯一值（或全部缺失），这项在后期的特征工程中可以丢弃，不作为后期机器学习建模的变量。

## 2. 特征的数值类型分析和可视化
特征一般都是由类别型特征和数值型特征组成，而数值型特征又分为连续型和离散型。
类别型特征有时具有非数值关系，有时也具有数值关系。比如‘grade’中的等级A，B，C等，是否只是单纯的分类，还是A优于其他要结合业务判断。
数值型特征本是可以直接入模的，但往往风控人员要对其做分箱，转化为WOE编码进而做标准评分卡等操作。从模型效果上来看，特征分箱主要是为了降低变量的复杂性，减少变量噪音对模型的影响，提高自变量和因变量的相关度。从而使模型更加稳定。
在开始展开分析工作前，我们要在心中明确一下分析的目的： 查找挖掘目标变量贷款违约（isDefault)和其他变量的关系。因而有必要先知道目标值的分布情况。

# 查看目标变量isDefault的分布
```python
train.isDefault.value_counts()
```
```
Out[32]:
0    640390
1    159610
Name: isDefault, dtype: int64
```
取值为0为否，即没有违约，为1是违约客户
也可以加入normalize参数，查看百分比：
```py
train.isDefault.value_counts(normalize=True)
0    0.800488
1    0.199513
Name: isDefault, dtype: float64
```
即80.05%的客户没有出现违约，而剩下19.95%（共159610）个客户违约。






## 数值连续型特征概率分布可视化
对于连续型变量，可以通过变量的概率密度分布图查看变量的大体分布情况
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100521351416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)

做这一步的目的是查看某一个数值型变量的分布，观察该变量是否符合正态分布，如果不符合正太分布的变量可以log化后再观察下是否符合正态分布。
**正态化的原因**：一些情况下正态非正态可以让模型更快的收敛，一些模型要求数据正态（eg. GMM、KNN）,保证数据不要过偏态即可，过于偏态可能会影响模型预测结果。

在实际操作中，除了可以用概率密度分布来查看某一个变量的数据分布外，还常用箱线图来查看数据分布集中度。
通过观察，可以看到 'loanAmnt', 'interestRate', 'installment', 'postCode', 'regionCode', ' 'openAcc', 'totalAcc', 'n2', 'n3', 等变量分布类似于正态分布，选择这些变量进一步用箱线图查看分布
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201005213655851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)
可以发现贷款金额（loanAmt)和贷款利率（'interestRate）较高时出现贷款违约的风险更高，即isDefault(目标变量）值为1。其他特征未发现明显区别。


通过各个简单的统计量来对数据整体的了解，分析各个类型变量相互之间的关系，以及用合适的图形可视化出来直观观察。通过对数据进行探索性分析，我们初步了解数据的分布情况，熟悉数据。这项工作为特征工程做准备的阶段。数据分析之所以重要，是因为很多时候该阶段提取出来的特征可以直接当作规则来用。
