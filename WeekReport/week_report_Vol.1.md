# 周报-Vol.1

### 1. 导入数据初步查看
```python
import numpy as np
import pandas as pd
train = pd.read_csv('./train.csv')
test = pd.read_csv('./testA.csv')
print('train data shape is: ', train.shape)
print('test data shape is: ', test.shape)`
```
> train data shape is:  (800000, 47)
>test data shape is:  (200000, 46)

### 2. 字段信息查看
```python
print(train.columns)
```
>Index(['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade',
       'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
       'annualIncome', 'verificationStatus', 'issueDate', 'isDefault',
       'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years',
       'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
       'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
       'initialListStatus', 'applicationType', 'earliesCreditLine', 'title',
       'policyCode', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
       'n9', 'n10', 'n11', 'n12', 'n13', 'n14'],
      dtype='object')
      
字段表
| Field |Description |
|--|--|
 id|为贷款清单分配的唯一信用证标识
loanAmnt|贷款金额
term	|贷款期限（year）
interestRate|	贷款利率
installment|	分期付款金额
grade	|贷款等级
subGrade|	贷款等级之子级
employmentTitle|	就业职称
employmentLength|	就业年限（年）
homeOwnership	|借款人在登记时提供的房屋所有权状况
annualIncome	|年收入
verificationStatus|	验证状态
issueDate	|贷款发放的月份
purpose	|借款人在贷款申请时的贷款用途类别
postCode|	借款人在贷款申请中提供的邮政编码的前3位数字
regionCode|	地区编码
dti	|债务收入比
delinquency_2years|	借款人过去2年信用档案中逾期30天以上的违约事件数
ficoRangeLow	|借款人在贷款发放时的fico所属的下限范围
ficoRangeHigh|	借款人在贷款发放时的fico所属的上限范围
openAcc	|借款人信用档案中未结信用额度的数量
pubRec	|贬损公共记录的数量
pubRecBankruptcies|	公开记录清除的数量
revolBal	|信贷周转余额合计
revolUtil|	循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
totalAcc|	借款人信用档案中当前的信用额度总数
initialListStatus|	贷款的初始列表状态
applicationType|	表明贷款是个人申请还是与两个共同借款人的联合申请
earliesCreditLine|	借款人最早报告的信用额度开立的月份
title	|借款人提供的贷款名称
policyCode|	公开可用的策略_代码=1新产品不公开可用的策略_代码=2
n系列匿名特征|	匿名特征n0-n14，为一些贷款人行为计数特征的处理| 1 |

### 3. 缺省值查看
```python
train.info()
```
```
RangeIndex: 800000 entries, 0 to 799999
Data columns (total 47 columns):
id                    800000 non-null int64
loanAmnt              800000 non-null float64
term                  800000 non-null int64
interestRate          800000 non-null float64
installment           800000 non-null float64
grade                 800000 non-null object
subGrade              800000 non-null object
employmentTitle       799999 non-null float64
employmentLength      753201 non-null object
homeOwnership         800000 non-null int64
annualIncome          800000 non-null float64
verificationStatus    800000 non-null int64
issueDate             800000 non-null object
isDefault             800000 non-null int64
purpose               800000 non-null int64
postCode              799999 non-null float64
regionCode            800000 non-null int64
dti                   799761 non-null float64
delinquency_2years    800000 non-null float64
ficoRangeLow          800000 non-null float64
ficoRangeHigh         800000 non-null float64
openAcc               800000 non-null float64
pubRec                800000 non-null float64
pubRecBankruptcies    799595 non-null float64
revolBal              800000 non-null float64
revolUtil             799469 non-null float64
totalAcc              800000 non-null float64
initialListStatus     800000 non-null int64
applicationType       800000 non-null int64
earliesCreditLine     800000 non-null object
title                 799999 non-null float64
policyCode            800000 non-null float64
n0                    759730 non-null float64
n1                    759730 non-null float64
n2                    759730 non-null float64
n3                    759730 non-null float64
n4                    766761 non-null float64
n5                    759730 non-null float64
n6                    759730 non-null float64
n7                    759730 non-null float64
n8                    759729 non-null float64
n9                    759730 non-null float64
n10                   766761 non-null float64
n11                   730248 non-null float64
n12                   759730 non-null float64
n13                   759730 non-null float64
n14                   759730 non-null float64
dtypes: float64(33), int64(9), object(5)
memory usage: 286.9+ MB
```
### 4. 赛题评价指标
本次竞赛采用AUC作为评价指标。
近年来，随着机器学习的相关技术从实验室走向实际应用，一些实际的问题对度量标准提出了新的需求。特别的，现实中样本在不同类别上的不均衡分布(class distribution imbalance problem)，使得分类精度这样的传统的度量标准不能恰当的反应分类器的“好坏”。
举个例子：假设一个100个待分类人群中有90个阴性和10个阳性病人。现在有一些分类器A、B对这个样本集进行分类。A分类器把这100个人全部当成了阴性，B分类器找出了10个阳性中的5个，剩下的95个全部当成了阴性。我们可以简单计算出A、B分类器的精度（Accury)都是90%。但是很明显，B分类器的performance要由于A。
另外，在一些分类问题中犯不同的错误代价是不同的(cost sensitive learning)。比如上面的例子中A分类器将阳性错分成阴性其后果要比阴性错分成阳性要高的多。

为了解决上述问题，人们从医疗分析领域引入了一种新的分类模型performance评判方法——ROC分析。

ROC的全名叫做Receiver Operating Characteristic，其主要分析工具是一个画在二维平面上的曲线——ROC curve。平面的横坐标是false positive rate(FPR)，纵坐标是true positive rate(TPR)。对某个分类器而言，我们可以根据其在测试样本上的表现得到一个TPR和FPR点对。这样，此分类器就可以映射成ROC平面上的一个点。调整这个分类器分类时候使用的阈值，我们就可以得到一个经过(0, 0)，(1, 1)的曲线，这就是此分类器的ROC曲线。如下图：


![image.png](https://img-blog.csdnimg.cn/20200928222312866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)


ROC曲线其实是多个混淆矩阵的结果组合。一般情况下，这个曲线都应该处于(0, 0)和(1, 1)连线的上方。因为(0, 0)和(1, 1)连线形成的ROC曲线实际上代表的是一个随机分类器。如果很不幸，你得到一个位于此直线下方的分类器的话，一个直观的补救办法就是把所有的预测结果反向，即：分类器输出结果为正类，则最终分类的结果为负类，反之，则为正类。虽然，用ROC curve来表示分类器的performance很直观好用。可是，人们总是希望能有一个数值来标志分类器的好坏。于是Area Under roc Curve(AUC)就出现了。顾名思义，AUC的值就是处于ROC curve下方的那部分面积的大小。通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的performance。如下图：


![image.png](https://img-blog.csdnimg.cn/2020092822234111.png#pic_center)

1） AUC = 1，是完美分类器（上图中的左图），绝大多数预测的场合，不存在完美分类器。
2） 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值（上图中间的图）。
3）AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值（上图中的左图）。
4）AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。
