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


class SilhouetteSuite extends SparkFunSuite with TestSuiteBase{

  test("Silhouette evaluation of iterative K-Means") {

    val sc = new SparkContext(conf)
    val data = sc.parallelize(Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(9.0, 9.0, 9.0),
      Vectors.dense(11.0, 11.0, 11.0)
    ))

    // obtain a model from iterative K-Means
    val numClusters = 2
    val iterations = 10
    val kMeansModel = KMeans.train(data, numClusters, iterations)

    // obtain the silhouette results
    val iterativeKMeansResults = new Silhouette(kMeansModel, data).runAnalysis()
    sc.stop()

    assert(iterativeKMeansResults(0)._2 == iterativeKMeansResults(1)._2)
    assert(iterativeKMeansResults(0)._3 == iterativeKMeansResults(1)._3)

  }

  test("Silhouette evaluation of Online K-Means") {

    val sc = new SparkContext(conf)
    val data = sc.parallelize(Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    ))

    // obtain a model from online K-Means
    val numClusters = 2
    val seed = 1
    val onlineKMeansModel = OnlineKMeans.train(data, numClusters, seed)

    // obtain the silhouette results
    val onlineKMeansResults = new Silhouette(onlineKMeansModel, data).runAnalysis()
    sc.stop()

    assert(onlineKMeansResults(0)._2 == onlineKMeansResults(1)._2)
    assert(onlineKMeansResults(0)._3 == onlineKMeansResults(1)._3)

  }

  test("Silhouette evaluation of Streaming Online K-Means") {

    val k = 2
    val seed = 1
    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    // obtain a model from Streaming Online K-Means
    val model = new StreamingOnlineKMeans(k, seed)

    val scc: StreamingContext = setupStreams(Seq(data), (inputDStream: DStream[Vector]) => {
      model.trainOn(inputDStream)
      inputDStream.count()
    })

    runStreams(scc, 1, 1)
    val result = model.latestModel()

    // obtain the silhouette results
    val sc = new SparkContext(conf)
    val parallelData = sc.parallelize(data)
    val streamingOnlineKMeansResults = new Silhouette(result, parallelData).runAnalysis()
    sc.stop()

    assert(streamingOnlineKMeansResults(0)._2 == streamingOnlineKMeansResults(1)._2)
    assert(streamingOnlineKMeansResults(0)._3 == streamingOnlineKMeansResults(1)._3)

  }

  test("Silhouette evaluation of ART K-Means") {

    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    val percentageOfSpace = 0.3
    val model = new ARTKMeans(percentageOfSpace)

    val scc: StreamingContext = setupStreams(Seq(data), (inputDStream: DStream[Vector]) => {
      model.trainOn(inputDStream)
      inputDStream.count()
    })

    runStreams(scc, 1, 1)
    val result = model.latestModel()

    // obtain the silhouette results
    val sc = new SparkContext(conf)
    val parallelData = sc.parallelize(data)
    val artKMeansResults = new Silhouette(result, parallelData).runAnalysis()
    sc.stop()

    assert(artKMeansResults(0)._2 == artKMeansResults(1)._2)
    assert(artKMeansResults(0)._3 == artKMeansResults(1)._3)

  }

}
