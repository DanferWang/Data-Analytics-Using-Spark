 # 利用决策树预测森林植被
1.	回归
	思想：从“某些值”预测“另外某个值”
	回归和分类关系紧密，回归是预测一个数值型数量，分类是预测标号或类别，都属于监督型学习
	决策树算法、随即决策森林算法，既可以做回归问题，也可以做分类问题
2.	向量与特征
特征，维度，预测指标，变量：一个东西，可以被量化
特征值按照顺序排列，就是特征向量
特征可以被简单的分为：类别型特征（分类问题，离散值中选取）、数值型特征（回归问题，可用数值量化，排序有意义）
3.	样本训练
学习算法需要在大量数据上训练，需要大量正确的输入和相对的输出
训练样本（训练集）=结构化的输入+输出目标
有些算法只能处理数值型特征
4.	决策树与决策森林
单棵决策树可以并行构建，许多决策树可以同时并行构建
选择抉择规则、停止抉择过程
要防止对训练样本的过度拟合
5.	Covtype数据集
文件：https://archive.ics.uci.edu/ml/datasets/Covertype
该数据集记录了美国科罗拉多州不同地区的森林植被类型
54个特征，1个标签，581012条样本
6.	准备数据
把数据解析成DataFrame，没有表头
根据数据集info，对54个特征的编码方式进行分析，给上述DataFrame加入列名，
目标列需要转换成Double型，Spark MLlib所要求
因为maven升级的时候没有自动加载完整的依赖包，csv文件中格式有特殊要求，因此在pom中添加commons-lang3依赖：
<dependency>
      <groupId>org.apache.commons</groupId>
      <artifactId>commons-lang3</artifactId>
      <version>3.5</version>
</dependency>
7.	第一棵决策树
评价指标采用准确率
90%训练集带有交叉验证，10%测试集
对数据集预处理：将除了预测目标列外的特征作为特征向量，使用VectorAssembler类
特征向量以稀疏向量的方式储存
transform“管道”技术
构建决策树：
model = new DecisionTreeClassifier()
	.setSeed(Random.nextLong())
	.setLabelCol(“Cover_Type”)
	.setFeatureCol(“featureVector”)
	.setPredictionCol(“prediction”)
	.fit(assembledTrainData)
	检查模型训练结果：model.toDebugString
	评估每个特征对正确预测的贡献度
	在训练集上比较预测值与正确目标
	使用MulticlassClassificationEvaluation对模型和输出做评价：准确率、F1
	混淆矩阵：正确预测仅在对角线，其余都为错误预测
	定义概率猜测“分类器”以计算随机猜测准确率，并于决策树模型准确率比较
	发现：决策树模型的准确率明显要大于随机猜测的准确率，尽管在没调参的情况下，有大量的错误预测，也要比随机的好
8.	决策树的超级参数
最大深度：决策树的层数，也是用决策树判断的最大次数，限制该参数有利于防止过拟合
bin：决策规则集合，bin的数量越多，所需处理时长越长，但找到的决策规则更优
不纯性度量：某个规则能清楚地把一些类别和其他类别分开。Gini不纯度（Spark默认）、熵
最小信息增益，可使最小不纯度降低，有利避免模型过拟合
9.	决策树调优
pipeline“管道”技术，适合目标数据集结构复杂，需要多次处理，或是在学习过程中要使用多个transform和estimate的情况，以帮助创建和调试实际的机器学习工作流
创建两个Transformer操作 VectorAssembler和DecisionTreeClassifier，构成一个Pipeline对象
使用ParamGridBuilder测试超参数组合
使用TrainValidationSplit将这些组件形成管道
	检查所有超参数组合的准确率，提取最佳超参组合
	比较最佳超参组合下，验证集与测试集模型准确率
10.	解码类别型特征
当前所有的特征都是按照数值型特征输入到决策树中的，输入中明显存在类别型特征，只是我们用二进制编码方式，将其转换为数值型
解码类别型特征可以减少内存使用量，提高处理速度
定义解码函数，UDF（用户自定义函数）
需要添加这个库: import org.apache.spark.mllib.linalg.Vectors
11.	随机决策森林
构建多棵决策树的方法注入了随机因素，可有效防止过拟合的发生
对所有决策树预测加权平均
只需使用RandomForestClassifier代替DecisionTreeClassifier，后面的处理方式、API都相同
12.	预测
删除测试集中的Cover_Type列，进行一次预测
