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
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext

class SilhouetteSuite extends SparkFunSuite with MLlibTestSparkContext{

  test("Silhouette model comparison") {

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

    // obtain the silhouette results for both forms of clustering
    val onlineKMeansResults = new Silhouette(onlineKMeansModel, data).runAnalysis()

    assert(onlineKMeansResults(0)._2 == onlineKMeansResults(1)._2)
    assert(onlineKMeansResults(0)._3 == onlineKMeansResults(1)._3)

  }

}
