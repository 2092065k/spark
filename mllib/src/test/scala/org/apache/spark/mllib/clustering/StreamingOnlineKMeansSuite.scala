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

class StreamingOnlineKMeansSuite extends SparkFunSuite with TestSuiteBase {

  test("Base case Streaming Online K-means clustering") {

    val k = 2
    val seed = 1
    val data = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(2.0, 2.0, 2.0),
      Vectors.dense(8.0, 8.0, 8.0),
      Vectors.dense(10.0, 10.0, 10.0)
    )

    val model = new StreamingOnlineKMeans(k, seed)

    val scc: StreamingContext = setupStreams(Seq(data), (inputDStream: DStream[Vector]) => {
      model.trainOn(inputDStream)
      inputDStream.count()
    })

    runStreams(scc, 1, 1)
    val finalCenters = model.latestModel().clusterCenters

    val firstCenter = Vectors.dense(9.0, 9.0, 9.0)
    val secondCenter = Vectors.dense(1.0, 1.0, 1.0)

    assert(finalCenters.length === 2)
    assert(finalCenters(0) ~== firstCenter absTol 1E-5)
    assert(finalCenters(1) ~== secondCenter absTol 1E-5)
  }

}
