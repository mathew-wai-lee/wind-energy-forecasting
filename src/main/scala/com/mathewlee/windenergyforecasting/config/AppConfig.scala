// package com.mathewlee.windenergyforecasting

// import pureconfig._
// import pureconfig.generic.auto._

// case class JobConfig(
//   inputPath: String,
//   outputPath: String,
//   threshold: Double
// )

// object AppConfig {
//   def loadOrThrow(configPath: String = "job"): JobConfig = {
//     ConfigSource.default.at(configPath).loadOrThrow[JobConfig]
//   }

//   val conf = new SparkConf()
//     .setMaster("local")
//     .setAppName("my awesome app")
//   println("testing app with input: " + inputFile + " and output: " + outputFile)

// }
