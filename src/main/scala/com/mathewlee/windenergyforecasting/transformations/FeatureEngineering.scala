package com.mathewlee.windenergyforecasting

import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.lit

// src/main/scala/functions/Transformations.scala
object Transformations {
  // Pure function: DataFrame in, DataFrame out
  def addConstantColumn(df: DataFrame, colName: String, value: Int): DataFrame = {
    df.withColumn(colName, lit(value))
  }

//   // Pure function with type safety
//   def filterByThreshold[T : Numeric](ds: Dataset[T], threshold: T): Dataset[T] = {
//     import ds.sparkSession.implicits._
//     ds.filter(_ > threshold)
//   }
}