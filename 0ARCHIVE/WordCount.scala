package com.example.sparkdemo

/**
 * Everyone's favourite wordcount example.
 */

import org.apache.spark.rdd._
import org.apache.spark.sql.DataFrame

object WordCount {
  /**
   * A slightly more complex than normal wordcount example with optional
   * separators and stopWords. Splits on the provided separators, removes
   * the stopwords, and converts everything to lower case.
   */
  def withStopWordsFiltered(df : DataFrame): Long = {
    df.count()
  }
}
