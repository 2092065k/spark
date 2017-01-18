/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.{SparkContext, SparkFunSuite}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.streaming.{StreamingContext, TestSuiteBase}
import org.apache.spark.streaming.dstream.DStream

class SOMKMeansSuite extends SparkFunSuite with TestSuiteBase{

  test("Base case SOM K-means clustering") {

    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    val NNDimensions = 5
    val nSize = 1
    val sigma = 1.0
    val learningRate = 0.2
    val seed = 1

    val model = new SOMKMeans(NNDimensions, nSize, sigma, learningRate, seed)

    val scc: StreamingContext = setupStreams(Seq(data), (inputDStream: DStream[Vector]) => {
      model.trainOn(inputDStream)
      inputDStream.count()
    })

    runStreams(scc, 1, 1)
    val finalCenters = model.latestModel().clusterCenters

    assert(finalCenters.length === 25)

  }

  test("Test the sequential version of SOM K-Means") {

    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    val NNDimensions = 5
    val nSize = 1
    val sigma = 1.0
    val learningRate = 0.2
    val seed = 1

    val sc = new SparkContext(conf)
    val parallelData = sc.parallelize(data)
    val seqComp = new SOMKMeans(NNDimensions, nSize, sigma, learningRate, seed)
    seqComp.computeSequentially(parallelData.map(elem => (0, elem)))
    val seqModelClusters = seqComp.latestModel().clusterCenters
    sc.stop()

    assert(seqModelClusters.length === 25)

  }

  test("Test SOM K-Means with unbiased sampling") {

    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    val NNDimensions = 5
    val nSize = 1
    val sigma = 1.0
    val learningRate = 0.2
    val seed = 1
    val samplingSeed = 10

    val sc = new SparkContext(conf)
    val parallelData = sc.parallelize(data)
    val unbiasedComp = new SOMKMeans(NNDimensions, nSize, sigma, learningRate, seed)
    unbiasedComp.computeUnbiasedSampling(parallelData, samplingSeed)
    val unbiasedCompClusters = unbiasedComp.latestModel().clusterCenters
    sc.stop()

    assert(unbiasedCompClusters.length === 25)

  }

}
