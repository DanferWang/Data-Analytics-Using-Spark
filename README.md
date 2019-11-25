# Data Analytics Using Spark
This repository stores some projects about data analytics which are all programmed by Scala using Spark

## Music Recommendation
      This project is aimmed at simple and brief recommended system on music. 
      The rating metric is the playcount of every user on every artist whose songs have been played before.
      I use collaborative filter and ALS to establish model. The detail algorithm has been uploaded to this proiect directory alongwith one paper which provides the theory basis.
      Datasets are open-scourced, from Audioscrobbler, which have main three text file and one ReadMe.txt.
      What needs to caution is the datasets this GitHub project is NOT complete due to the limitation of file size, below 25MB. Please take care!
      The full datasets you can download by reading DATASET_README where there is an URL and extract code.

    Also, I have one DLog for you to learn and follow, however, it is in Chinese!
    
## Decision tree and Random forest
      I choose decison tree and random forest algorithm mainly becuase it is easy to be understood!
      However, I find it pity that it has possibility to report error on UDF(user defined function). I don't really know why it occurs, for this the same code running on several different computers comes out differnt results, maybe running healthily, or going dump. I guess the reason might be the version of Spark. I developed the project on Spark 2.11. And it can be compiled, but when executing in Spark it occurs error. Some fellows who discover the problem are welcomed to teach me to correct or improve it! Thank you all!
      Please pay attention to changing the relative file path.
      
      I do also took down some notes while programming. Please refer to the DLog, and expand some core tech!
      DLog is in Chinese, forgiving my poor English.
