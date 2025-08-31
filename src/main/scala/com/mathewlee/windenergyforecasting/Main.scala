package com.mathewlee.windenergyforecasting

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */
object mainProd extends App{
  // spark-submit command should supply all necessary config elements
  Runner.run(new SparkConf())
}


/**
  * Use this to test the app locally, from sbt:
  * sbt "run inputFile.txt outputFile.txt"
  *  (+ select CountingLocalApp when prompted)
  */
object mainLocal extends App{
  val conf = new SparkConf()
    .setMaster("local")
    .setAppName("test runner")
  Runner.run(conf)
}

object Runner {
  def run(conf: SparkConf): Unit = {
    val spark = SparkSession.builder()
      .config(conf)
      .getOrCreate()
    
    println("Spark Session created successfully.")
    
    val df = spark.read.csv("data/step1_original_merged_output.csv")
    println("CSV Read successfully")
    
    println("")
    println(df.show(5))
    println("")
    
    println("")
    println(df.columns)
    println("")
    
    println("")
    println(s"There are: ${df.count()} rows")
    println("")
  
    scala.io.StdIn.readLine()
  }
}

