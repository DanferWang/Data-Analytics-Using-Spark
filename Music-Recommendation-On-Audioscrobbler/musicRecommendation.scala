import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object musicRecommendation {
  def main(args: Array[String]): Unit = {
    //设置Spark参数
    val conf: SparkConf = new SparkConf().setAppName("MusicRecomendation").setMaster("local[*]")
    val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    //设置隐式转换
    import session.implicits._


    //数据准备
    //处理user_artist_data.txt
    val rawUserAritistData: Dataset[String] = session.read.textFile("Datasets/Audioscrobbler Music Recommendation/user_artist_data.txt")
    //测试载入
    // rawUserAritistData.take(5).foreach(println)

    //处理artist_data.txt
    val rawArtistData: Dataset[String] = session.read.textFile("Datasets/Audioscrobbler Music Recommendation/artist_data.txt")
    val artistByID: DataFrame = rawArtistData.flatMap { line =>
      var (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      }
      else {
        try {
          Some((id.toInt, name.trim))
        }
        catch {
          case _: NumberFormatException => None
        }
      }
    }.toDF("id", "name")
    //测试数据加载
    //artistByID.take(5).foreach(println)

    //处理artist_alias.txt
    val rawArtistAlias: Dataset[String] = session.read.textFile("Datasets/Audioscrobbler Music Recommendation/artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap { line =>
      var Array(artist, alias) = line.split("\t")
      try {

        if (artist.isEmpty) {
          None
        }
        else {
          Some((artist.toInt, alias.toInt))
        }
      }
      catch {
        case _: NumberFormatException => None
      }
    }.collect().toMap
    //测试数据加载
    //println(artistAlias.head)
    //artistByID.filter($"id" isin (1208690,1003926)).show()


    //构建模型
    //艺术家真名对应
    var bArtistAlias = session.sparkContext.broadcast(artistAlias)
    var trainData = buildCounts(session,rawUserAritistData, bArtistAlias)
    trainData.cache()

//    //构建ALS模型
//    val model: ALSModel = new ALS()
//      .setSeed(Random.nextLong()) // 使用随机种子
//      .setImplicitPrefs(true)
//      .setRank(10)
//      .setRegParam(0.01)
//      .setAlpha(1.0)
//      .setMaxIter(5)
//      .setUserCol("user")
//      .setItemCol("artist")
//      .setRatingCol("count")
//      .setPredictionCol("prediction")
//      .fit(trainData)
//
//    // 截取查看特征向量
//    model.userFactors.show(5, truncate = false) //第一个参数是查看用户数量
//
//
//    //检查推荐结果
//    //确定测试目标用户，检查播放过的艺术家
//    val userID = 2093760
//    val existingArtistIDs: Array[Int] = trainData.filter($"user" === userID).select("artist").as[Int].collect() //找到并收集该用户听过的艺术家ID
//    artistByID.filter($"id" isin (existingArtistIDs:_*)).show() //艺术家ID对应名字
//    //对目标用户进行推荐
//    val topRecommendations: DataFrame = makeRecommendation(session,model,userID,5)
//    val recommendedArtistID: Array[Int] = topRecommendations.select("artist").as[Int].collect()
//    artistByID.filter($"id" isin(recommendedArtistID:_*)).show()

//    //推荐质量评价
    //划分训练集和验证集
    val allData: DataFrame = buildCounts(session,rawUserAritistData,bArtistAlias)
    val Array(trainSet,cvSet): Array[Dataset[Row]] = allData.randomSplit(Array(0.9,0.1))
    trainSet.cache()
    cvSet.cache()

    //对艺术家ID去重
    val allArtistIDs: Array[Int] = allData.select("artist").as[Int].distinct().collect()
    val bAllArtistIDs: Broadcast[Array[Int]] = session.sparkContext.broadcast(allArtistIDs)

    //构建模型
    //相对较优超参值：rank=40，regParam=5.0，alpha=50.0
    val meanALSModel: ALSModel = new ALS()
      .setSeed(Random.nextLong())
      .setImplicitPrefs(true)
      .setRank(40)
      .setRegParam(5.0)
      .setAlpha(50.0)
      .setMaxIter(5)
      .setUserCol("user")
      .setItemCol("artist")
      .setRatingCol("count")
      .setPredictionCol("prediction")
      .fit(trainSet)

    //计算平均AUC
    var modelMeanAUC = meanAUC(session,cvSet,bAllArtistIDs,meanALSModel.transform)
    println("this model no mean AUC is "+ modelMeanAUC)


    //选择调整超参数
//    val evalutionParam =  // 相对较优超参值：rank=40，regParam=5.0，alpha=50.0
//    // 设置预选比较参数
//      for (rank <- Seq(30,40);
//           regParam <- Seq(4.0,5.0);
//           alpha <- Seq(40.0,50.0))
//        yield
//        {
//          val model = new ALS()
//            .setSeed(Random.nextLong())
//            .setImplicitPrefs(true)
//            .setRank(rank)
//            .setRegParam(regParam)
//            .setAlpha(alpha)
//            .setMaxIter(20)
//            .setUserCol("user")
//            .setItemCol("artist")
//            .setRatingCol("count")
//            .setPredictionCol("prediction")
//            .fit(trainSet)
//
//          val aucMean: Double = meanAUC(session, cvSet, bAllArtistIDs, model.transform)
//
//          // 释放资源
//          model.userFactors.unpersist()
//          model.itemFactors.unpersist()
//
//          //返回平均AUC及所属超参数
//          (aucMean, (rank, regParam, alpha))
//        }
//
//    //按照平均AUC排序输出
//    evalutionParam.sorted.reverse.foreach(println)

    val someUsers = Array(2063431,2006581,2147264,2304881,2407638)
    val someRecommendation = someUsers.map(userID => (userID,makeRecommendation(session,meanALSModel,userID,5)))
    session.conf.set("spark.sql.crossJoin.enabled","true")
    someRecommendation.foreach{ case (userID , recsDF) =>
      val recommendedArtist = recsDF.select("artist").as[Int].collect()
      println(s"for $userID recommend ${recommendedArtist.mkString(".")} ")
        // isin带的参数并不是list，而是一个可变长vararg
        artistByID.filter($"id" isin (recommendedArtist:_*)).show()
        println()
    }

  }

  //定义艺术家真名对应函数
  def buildCounts(sparkSession: SparkSession,rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    //设置隐式转换
    import sparkSession.implicits._

    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(s => s.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID.toInt, artistID.toInt) //如果艺术家存在别名，取艺术家别名，否则取原始名字
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
  }

  //定义推荐函数
  def makeRecommendation(sparkSession: SparkSession,model: ALSModel,userID:Int,amount:Int): DataFrame ={
    //设置隐式转换
    import sparkSession.implicits._

    val toRecommend = model.itemFactors.select($"id".as("artist")).withColumn("user",lit(userID)) //将所有艺术家ID与目标用户ID对应起来
    //利用训练好的模型，对新输入的数据（艺术家-目标用户）评分，返回最高的所需数量
    model.transform(toRecommend).select("artist","prediction").orderBy($"prediction".desc).limit(amount)
  }

  //定义平均AUC函数
  def meanAUC(session: SparkSession,positiveData : DataFrame,bAllArtistIDs : Broadcast[Array[Int]],predictFunction : (DataFrame => DataFrame)) : Double = {
    //隐式转换
    import session.implicits._

    //正例
    val positivePredictions: DataFrame = predictFunction(positiveData.select("user","artist")).withColumnRenamed("prediction","positivePrediction")
    //反例
    val negativeData = positiveData.select("user","artist").as[(Int,Int)].groupByKey{ case (user,_)=>user}.
      flatMapGroups{ case (userID,userIDAndPosArtistIDs)=>
        val random: Random = new Random()
        val posItemIDSet: Set[Int] = userIDAndPosArtistIDs.map{ case (_, artist)=>artist}.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs: mutable.ArrayOps[Int] = bAllArtistIDs.value
        var i = 0
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size){
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          if(!posItemIDSet.contains(artistID)){
            negative += artistID
          }
          i += 1
        }
        negative.map(artistID => (userID,artistID))
    }.toDF("user","artist")

    val negativePredictions = predictFunction(negativeData).withColumnRenamed("prediction","negativePrediction")
    val joinedPredictions = positivePredictions.join(negativePredictions,"user").select("user","positivePrediction","negativePrediction").cache()

    val allCounts = joinedPredictions.groupBy("user").agg(count(lit("1")).as("total")).select("user","total")
    val correctCounts = joinedPredictions.filter($"positivePrediction">$"negativePrediction").groupBy("user").agg(count("user").as("correct")).select("user","correct")

    val meanAUC: Double = allCounts.join(correctCounts,Seq("user"),"left_outer").select($"user",(coalesce($"correct",lit(0))/$"total").as("AUC")).agg(mean("AUC")).as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }


}