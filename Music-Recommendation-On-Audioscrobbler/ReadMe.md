# Audioscrobbler数据集上音乐推荐
## 数据集
	覆盖更多的用户和艺术家，也包含更多的总体信息。用户和艺术家的关系是通过其他行动隐含体现出来的，而不是通过显示的评分或者点赞得到的，这种类型的数据通常被称作隐式反馈数据。
## 交替最小二乘推荐算法
  	适合隐式反馈数据，协同过滤算法
  	数据集稀疏
	算法扩展性好，速度快
	潜在因素模型：通过相对少量的未被观察到的底层原因，来解释大量用户和产品之间可观察到的交互。
## 矩阵分解模型
	交替最小二乘（ALS）：Collaborative Filtering for Implicit Feedback Datasets、Large-scale Parallel Collaborative Filtering for the Netflix Prize
	最小二乘：已知Y可以得到X，X可以表达成Y和A的函数，想要精确相等不可能，因此转化成最小化
	借助QR分解求矩阵的逆
	交替：由X计算Y，再由Y计算X，如此反复
	并行化
## 数据准备
	三个生数据文件：user_artist_data.txt、artist_data.txt、artist_alias.txt
	为了保证ALS执行效率，需要将数据设置为Int类型，这就要保证所有数据要在Int类型合法的范围内
### user_artist_data.txt
	一个用户ID、一个艺术家ID、播放次数，用“ ”分隔，处理成DataFrame: userId[Int]、artistId[Int]
### artist_data.txt
	艺术家ID与艺术家名字的对应，用“\t”分割，但是该数据文件不够干净，会出现少量非法信息，需要使用flatMap以及Option类：Some、None，处理成DataFrame：id[Int]、name[String]
### artist_alias.txt
	id、id，将拼写错误或者有二意性的艺术家名字映射到正规唯一名字，该数据文件也有一点小问题，部分映射缺失第一个id，需要被过滤掉，处理成Map(artistId[Int], aliasId[Int])
## 构建模型
### 艺术家真名对应
	用别名数据集（artist_alias.txt）将所有艺术家ID转换成正规ID，如果艺术家存在别名，取得其对应真名，否则取得其原始名字。
	对artistAlias使用广播变量，对每一次map操作不必要充分副本发送，直接广播到内存及executor。
	调用cache，表示计算完成后，将结果暂时保存在内存，因为ALS算法是迭代的，所以在内存中缓存可以减少通信量。
### 构建ALS模型
	new ALS().
	setSeed() //随机初始向量
	setImplicitPrefs() //是否为隐式数据
	setRank() //模型潜在因素（特征）个数，一般也是矩阵的阶
	setAlpha() //控制矩阵分解时，观察到与没观察到的权重
	setMaxIter() //矩阵分解迭代次数
	setRegParam() //标准过拟合参数
	setUserCol() //用户
	setItemCol() //产品
	setRatingCol() //评分
	setPredictionCol() //预测
	fit() //输入数据集

	model.userFactors.show(1,truncate = false) //截取特征向量
## 检查推荐结果
	确定测试用户，检查播放过的艺术家，提取艺术家ID及其对应艺术家名字。
	定义推荐函数：选择所有艺术家ID与对应的目标用户ID，对所有艺术家评分，返回分值最高的。
	不必过虑目标用户已经听过的艺术家ID。
	计算耗时大，适用于批量评分推荐，不适合实时评分。
	得到艺术家ID后，可以根据映射关系，得到艺术家名字。
## 推荐质量评价
	很难确定推荐结果的好坏，因为对于用户的个性化推荐是通过训练好的协同矩阵以输入用户为对象给每个艺术家评分
	将原始数据分出一些播放数据作为优秀推荐验证数据，其余为训练集
	以此可以计算该推荐算法的得分：比较优秀推荐和算法推荐的占比
	ROC、AUC
	计算（平均）AUC(定义函数)
	划分数据为训练集（90%）和验证集（10%）
	K折交叉验证
	
## 调参、选参
	调整模型的rank、regParam、alpha，得到更大的AUC
	猜想几个超参数的值，组合后按照AUC从大到小的排序
	按照一定规律调整参数，达到期望预期效果
	发现相对有优势的超参数值

## 产生最终推荐
	目前是对单一用户进行推荐，每次起一个Spark任务只能对一个用户ID进行推荐，要达成的目标是可以输入一组用户同时对其推荐,map即可
	将ArtistID转换成艺术家名字，统一输出
	也可以向艺术家推荐用户，类似于发现相同兴趣的人，在解析输入的时候，兑换用户和艺术家的字段即可
