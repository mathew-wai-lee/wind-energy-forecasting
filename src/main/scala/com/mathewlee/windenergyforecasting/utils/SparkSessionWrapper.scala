// src/main/scala/utils/SparkSessionWrapper.scala
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

trait SparkSessionWrapper {
  def spark: SparkSession
}

trait DataLoader {
  self: SparkSessionWrapper =>
  
  // Generic loader method
  def loadData(path: String, format: String = "parquet"): DataFrame = {
    spark.read.format(format).load(path)
  }
}