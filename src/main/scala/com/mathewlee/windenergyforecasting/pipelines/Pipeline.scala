// class DataProcessor(config: JobConfig) 
//   extends SparkSessionWrapper with DataLoader {
  
//   import Transformations._

//   def run(): Unit = {
//     // Load data using generic loader
//     val input = loadData(config.inputPath)
    
//     // Use pure transformations
//     val transformed = addConstantColumn(input, "batch_id", 42)
//       .transform(df => filterByThreshold(df, config.threshold))
    
//     transformed.write.parquet(config.outputPath)
//   }
// }

// // SparkSession provider implementation
// trait ProductionSparkSession extends SparkSessionWrapper {
//   lazy val spark: SparkSession = SparkSession.builder
//     .appName("DataProcessor")
//     .getOrCreate()
// }
