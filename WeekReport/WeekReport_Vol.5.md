# 模型融合
通过融合多个不同的模型，可能提升机器学习的性能。
一般来说，模型之间差异越大，融合所得的结果将会更好。这种特性不会受融合方式的影响。注意这里所指模型之间的差异，是指模型之间相关性的差异。

一般经常用到的融合方法有：

 1. **简单方法**：</Br>
	平均法（Averaging）-针对回归问题</Br>
	投票法（Voting）- 针对分类问题
 2. **高阶方法**：</Br>
	Stacking</Br>
	Blending
	
**平均法Averaging**</Br>
平均法是针对回归问题而设计的模型融合方法。假如针对一个具体回归任务，设计了三种不同的模型，各自有着不同的预测结果。平均法就是将折三个不同的模型的预测结果进行简单平均，得到一个新的预测值来作为最终的结果。
通过在取平均值的过程，给予不同的模型不同的权重，平均法又可以分为简单平均和加权平均。前者是对不对的模型给予同样的权重。而后者会根据模型预测结果的准确率调整权重值，相当于给正确率高的模型更高的权重。
两者的公式如下：

简单加权</Br>
![https://math.jianshu.com/math?formula=H(x)%20%3D%20%5Cfrac%7B1%7D%7BT%7D%5Csum_%7Bi%3D1%7D%5ETh_i(x)](https://img-blog.csdnimg.cn/20201026185754640.png#pic_center)

加权平均</Br>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201026185827344.png#pic_center)

**投票Voting**</Br>
投票是一种非常朴素的方法，即少数服从多数。这种思想在随机方法中也有类似的应用。
对于分类问题，假设有三个相互独立的模型，每个正确率都是80%，采用少数服从多数的方式进行投票。那么最终的正确率将是：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102619024913.png#pic_center)

（注：前一种情况为三者均预测正确，概率为0.8x0.8x0.8，后者为三者中有两个模型预测正确，一个错误）
可以看到正确率由单个的0.8提升到了0.89。可以想象，如果进行投票的模型越多，那么显然其结果将会更好。但是其前提条件是模型之间相互独立，结果之间没有相关性。越相近的模型进行融合，融合效果也会越差。
比如对于一个正确输出全为1的测试，我们有三个很相近的的预测结果，分别为：</Br>


![image.png](https://img-blog.csdnimg.cn/2020102619031294.png#pic_center)

进行投票其结果为：</Br>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201026190323150.png#pic_center)

而假如我们的各个预测结果之间有很大差异：</Br>

![image.png](https://img-blog.csdnimg.cn/20201026190332405.png#pic_center)

其投票结果将为：</Br>
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102619034359.png#pic_center)


可见模型之间差异越大，融合所得的结果将会更好。

以分类任务为例，假设存在多个不同的模型，多个模型具有不同的分类结果。对于一个对象而言，最终的分类结果可以采用投票最多的类为最终的预测结果。

**Stacking**</Br>
Stacking模型的本质是一种分层的结构，用了大量的基分类器，将其预测的结果作为下一层输入的特征，这样的结构使得它比相互独立训练模型能够获得更多的特征。</Br>


![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102619043133.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)

如图：图中有五个基分类器，将数据分别放入其中进行训练然后得到预测结果，再将得到的五个预测结果作为模型六的输入特征再次进行训练，得到最终结果。但是由于直接由五个基学习器获得结果直接带入模型六中，容易导致过拟合，因此在使用五个及以上模型进行训练的时候，使用k折交叉验证。

**Blending**</Br>
blending是将预测的值作为新的特征和原特征合并，构成新的特征值，用于预测。为了防止过拟合，将数据分为两部分d1、d2，使用d1的数据作为训练集，d2数据作为测试集。预测得到的数据作为新特征使用d2的数据作为训练集结合新特征，预测测试集结果。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201026191602851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkZ2hqZ2Y=,size_16,color_FFFFFF,t_70#pic_center)

**Stacking**与**Blending**区别：</Br>
a. Stacking</Br>
（1）Stacking中由于两层使用的数据不同，所以可以避免信息泄露的问题。</Br>
（2）在组队竞赛的过程中，不需要给队友分享自己的随机种子。</Br>
b. Blending</Br>
（1）Blending比Stacking简单，不需要构建多层模型。</Br>
（2）由于Blending对将数据划分为两个部分，在最后预测时有部分数据信息将被忽略。</Br>
（3）同时在使用第二层数据时可能会因为第二层数据较少产生过拟合现象。</Br>



