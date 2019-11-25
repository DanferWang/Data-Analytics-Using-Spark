package com.Danfer.AdvancedAnalytics

import org.apache.spark.sql.functions._
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Model, Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.mllib.linalg.Vector

import scala.util.Random

object decisionTreeRandomForestPredict {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setAppName("Decision Tree Random Forest Prediction").setMaster("local[*]")
    val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    // 拒绝INFO
    session.sparkContext.setLogLevel("ERROR")
    // 隐式转换
    import session.implicits._

    // 准备数据
    // 解析数据
    val dataNoHeaderDF: DataFrame = session.read.option("inferSchema",true).option("header",false).csv("data/DecisionTree and RandomForest Prediction Covtype/covtype.data")
    // 根据编码方式，添加列名
    val colNames: Seq[String] = Seq("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am"
      ,"Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"
    )++((0 until 4).map(i => s"Wilderness_Area_$i")
      )++((0 until 40).map(i => s"Soil_Type_$i")
      )++Seq("Cover_Type")
    // 目标列转换成Double型
    val data: DataFrame = dataNoHeaderDF.toDF(colNames:_*).withColumn("Cover_Type", $"Cover_Type".cast("double"))
//    // 检查数据准备
//    println(data.head)


    //决策树 Decision Tree
    // 划分训练集、测试集
    val Array(trainData,testData): Array[Dataset[Row]] = data.randomSplit(Array(0.9,0.1))
    trainData.cache()
    testData.cache()
    // 数据预处理
    // 生成特征向量与预测目标列
    val inputCol: Array[String] = trainData.columns.filter(_ != "Cover_Type")
    val assembler: VectorAssembler = new VectorAssembler().setInputCols(inputCol).setOutputCol("featureVector")
    val assembledTrainData: DataFrame = assembler.transform(trainData)
//    // 查看生成特征向量效果
//    assembledTrainData.select("featureVector").show(truncate = false) // 只显示20条
    // 构建决策树
    val model: DecisionTreeClassificationModel = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")
      .fit(assembledTrainData)
//    // 检查模型训练结果
//    println(model.toDebugString)
//    // 评估每个特征对正确预测的贡献度
//    model.featureImportances.toArray.zip(inputCol).sorted.reverse.foreach(println)
//    // 在训练集上，比较预测值与正确目标
//    // 正确率很低！！
    val predict: DataFrame = model.transform(assembledTrainData)
//    predict.select("Cover_Type","prediction","probability").show(truncate = false)

//    // 评价该模型
//    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("Cover_Type")
//      .setPredictionCol("prediction")
//    // 准确率
//    println("accuracy is " + evaluator.setMetricName("accuracy").evaluate(predict))
//    // F1
//    println("F1 is " + evaluator.setMetricName("f1").evaluate(predict))
//    // 混淆矩阵
//    val predictRDD: RDD[(Double, Double)] = predict.select("prediction","Cover_Type").as[(Double,Double)].rdd
//    val predicConfusionMatrix = new MulticlassMetrics(predictRDD).confusionMatrix
//    println("The confusion matrix is ")
//    println(predicConfusionMatrix)

//     // 比较随机猜测的准确率
//        val trainProbability: Array[Double] = classProbabilities(session,trainData)
//        val testProbability: Array[Double] = classProbabilities(session,testData)
//        val randomProb: Double = trainProbability.zip(testProbability).map { case (trainProb, testProb) =>
//          trainProb * testProb
//        }.sum
//        println("The random probability is " + randomProb)

    // 创建Pipeline
    val classifier :DecisionTreeClassifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")
    val pipeline: Pipeline = new Pipeline().setStages(Array(assembler,classifier))

    // 测试超餐指标
    val paramGrid: Array[ParamMap] = new ParamGridBuilder() // 实际最佳超参数为：entropy、30、40、0.0
      .addGrid(classifier.impurity, Seq("gini", "entropy"))
      .addGrid(classifier.maxDepth, Seq(20,30))
      .addGrid(classifier.maxBins, Seq(30, 40))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.025))
      .build()
    val multiclassEval: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val validatorModel = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiclassEval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)
      .fit(trainData)
    // 提取最佳超参组合
    val bestModel: Model[_] = validatorModel.bestModel
//    val bestParamMap: ParamMap = bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap()
//    println("the best hyper-parameter : " + bestParamMap)
//    // 检查所有组合准确率
//    val paramAndMetrics: Array[(Double, ParamMap)] = validatorModel.validationMetrics.zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)
//    paramAndMetrics.foreach{case (metrics, param) =>
//      println(metrics)
//      println(param)
//      println()
//    }

    // 比较最佳超参组合下，验证集与测试集模型准确率
    println("in validation : " + validatorModel.validationMetrics.max)
    println("in test : " + multiclassEval.evaluate(bestModel.transform(testData)))

//    // 对训练集和测试集解码
//    val uncodeTrainData: DataFrame = uncode(session,trainData)
//    val uncodeTestData: DataFrame = uncode(session,testData)
//
//    val uncodeInputCols: Array[String] = uncodeTrainData.columns.filter(_ != "Cover_Type")
//    var uncodeAssembler = new VectorAssembler()
//      .setInputCols(uncodeInputCols)
//      .setOutputCol("uncodeFeatureVector")
//    var uncodeIndexer = new VectorIndexer()
//      .setMaxCategories(40)
//      .setInputCol("uncodeFeatureVector")
//      .setOutputCol("uncodeIndexedVector")
//    var uncodeClassifier = new DecisionTreeClassifier()
//      .setSeed(Random.nextLong())
//      .setLabelCol("Cover_Type")
//      .setFeaturesCol("uncodeIndexedVector")
//      .setPredictionCol("uncodePrediction")
//
//    var uncodePipeline = new Pipeline().setStages(Array(uncodeAssembler,uncodeIndexer,uncodeClassifier))
//
//    val uncodeParamGrid: Array[ParamMap] = new ParamGridBuilder() // 实际最佳超参数为：entropy、30、40、0.0
//      .addGrid(classifier.impurity, Seq("gini", "entropy"))
//      .addGrid(classifier.maxDepth, Seq(20,30))
//      .addGrid(classifier.maxBins, Seq(30, 40))
//      .addGrid(classifier.minInfoGain, Seq(0.0, 0.025))
//      .build()
//    val uncodeMulticlassEval: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("Cover_Type")
//      .setPredictionCol("uncodePrediction")
//      .setMetricName("accuracy")
//    val uncodeValidatorModel = new TrainValidationSplit()
//      .setSeed(Random.nextLong())
//      .setEstimator(uncodePipeline)
//      .setEvaluator(uncodeMulticlassEval)
//      .setEstimatorParamMaps(uncodeParamGrid)
//      .setTrainRatio(0.9)
//      .fit(uncodeTrainData)
//    // 提取最佳超参组合
//    val uncodeBestModel: Model[_] = uncodeValidatorModel.bestModel
//
//    // 比较最佳超参组合下，验证集与测试集模型准确率
//    println("uncode in validation : " + uncodeValidatorModel.validationMetrics.max)
//    println("uncode in test : " + uncodeMulticlassEval.evaluate(uncodeBestModel.transform(uncodeTestData)))

    // 构建随机森林
      var randomForestClassifier = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")
        .setMaxBins(40)
        .setMaxDepth(30)
        .setMinInfoGain(0.0)
        .setImpurity("entropy")
        .setNumTrees(25)

    var forestModel = randomForestClassifier.fit(assembledTrainData)

    val prediction: DataFrame = forestModel.transform(assembledTrainData)

    // 评价该模型
    val forestEvaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
    // 准确率
    println("the random forest accuracy is " + forestEvaluator.setMetricName("accuracy").evaluate(prediction))

    // 预测
    forestModel.transform(testData.drop("Cover_Type")).select("prediction").show()

  }

  // 定义猜测函数
  def classProbabilities(sparkSession: SparkSession,data : DataFrame): Array [Double] = {
    import sparkSession.implicits._
    val total: Long = data.count()
    data.groupBy("Cover_Type").count().orderBy("Cover_Type").select("count").as[Double].map(_ / total).collect()
  }

  // 定义类别型特征解码函数
  def uncode(sparkSession: SparkSession, data:DataFrame):DataFrame={
    import sparkSession.implicits._
    // 处理Wilderness_Area (4 binary columns)
    val wildernessCols: Array[String] = (0 until 4).map(i => s"Wilderness_Area_$i").toArray
    val wildernessAssembler: VectorAssembler = new VectorAssembler().setInputCols(wildernessCols).setOutputCol("wilderness")
    val uncodeUDF= udf((vec : Vector) => vec.toArray.indexOf(1.0).toDouble)
    val numWilderness: DataFrame = wildernessAssembler.transform(data).drop(wildernessCols:_*) // 删除不需要的原编码Wilderness_Area列
      .withColumn("wilderness", uncodeUDF($"wilderness")) // 用数字代表不同Wilderness_Area类别

    // 处理Soil_Type (40 binary columns)
    val soilCols: Array[String] = (0 until 40).map(i => s"Soil_Type_$i").toArray
    val soilAssembler: VectorAssembler = new VectorAssembler().setInputCols(soilCols).setOutputCol("soil")
    soilAssembler.transform(numWilderness)
      .drop(soilCols:_*) // 删除不需要的原编码Soil_Type列
      .withColumn("soil",uncodeUDF($"soil")) // 用数字代表不同Soil_Type类别
  }

}
