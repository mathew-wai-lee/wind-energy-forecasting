name := "Simple Project"

version := "1.0"

scalaVersion := "2.13.16"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.0.0",
  "org.apache.spark" %% "spark-sql" % "4.0.0",
  "org.apache.spark" %% "spark-hive" % "4.0.0",
  "org.scala-lang" %% "toolkit" % "0.7.0",
  "com.github.pureconfig" %% "pureconfig" % "0.17.9",
  "org.scalatest" %% "scalatest" % "3.2.15" % "test"
)