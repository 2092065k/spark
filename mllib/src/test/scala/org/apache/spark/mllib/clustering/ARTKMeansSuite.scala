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

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.streaming.{StreamingContext, TestSuiteBase}
import org.apache.spark.streaming.dstream.DStream

class ARTKMeansSuite extends SparkFunSuite with TestSuiteBase{

  test("Base case ART K-means clustering") {

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
    val finalCenters = model.latestModel().clusterCenters

    val firstCenter = Vectors.dense(1.0, 1.0, 1.0)
    val secondCenter = Vectors.dense(9.0, 9.0, 9.0)

    assert(finalCenters.length === 2)
    assert(finalCenters(0) ~== firstCenter absTol 1E-5)
    assert(finalCenters(1) ~== secondCenter absTol 1E-5)
  }

  test("ART K-means safely handles empty RDDs") {

    val model = new ARTKMeans(0.5)
    val point = Vectors.dense(0.0, 0.0, 0.0)

    val scc: StreamingContext = setupStreams(Seq(Seq(), Seq(point)),
      (inputDStream: DStream[Vector]) => {
        model.trainOn(inputDStream)
        inputDStream.count()
      })

    runStreams(scc, 2, 1)
    val finalCenters = model.latestModel().clusterCenters


    assert(finalCenters.length === 1)
    assert(finalCenters(0) ~== point absTol 1E-5)
  }

}
