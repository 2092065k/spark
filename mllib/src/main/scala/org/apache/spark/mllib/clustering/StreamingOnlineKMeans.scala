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

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.random.XORShiftRandom

class StreamingOnlineKMeansModel(
    override val clusterCenters: Array[Vector],
    val clusterWeights: Array[Double]) extends KMeansModel(clusterCenters) with Logging {

  def update(data: RDD[Vector]): StreamingOnlineKMeansModel = {

    val sc = data.sparkContext
    val closest = data.map(point => (this.predict(point), point))
    val dims = clusterCenters(0).size
    val bcCenters = sc.broadcast(clusterCenters)
    val bcWeights = sc.broadcast(clusterWeights)

    val final_centers_and_weights = closest.groupByKey().mapPartitions{ pointsInfo =>
      val localCenters = bcCenters.value
      val localWeights = bcWeights.value

      val centersValue = Array.fill(localCenters.length)(Vectors.zeros(dims))
      val counts = Array.fill(localWeights.length)(0.0)

      pointsInfo.foreach{ pointsPerCenterInfo =>
        val centerIndex = pointsPerCenterInfo._1
        val points = pointsPerCenterInfo._2
        counts(centerIndex) = localWeights(centerIndex)

        points.foreach{ point =>

          val center = localCenters(centerIndex)
          counts(centerIndex) += 1

          // add the contribution of one point
          val contrib = new DenseVector(Array.fill(dims)(0.0))
          axpy(1.0, point, contrib)
          axpy(-1.0, center, contrib)
          scal(1.0/counts(centerIndex), contrib)
          axpy(1.0, contrib, center)
          centersValue(centerIndex) = center
        }

      }
      counts.indices.filter(counts(_) > 0).map(j => (j, (centersValue(j), counts(j)))).iterator
    }.collect()

    final_centers_and_weights.foreach{ centerInfo =>
      val centerIndex = centerInfo._1
      val newCenterValue = centerInfo._2._1
      val newCenterWeight = centerInfo._2._2

      clusterWeights(centerIndex) = newCenterWeight
      clusterCenters(centerIndex) = newCenterValue
    }

    this
  }

}

class StreamingOnlineKMeans (
    var k: Int,
    var seed: Long)  extends Logging with Serializable {

  protected var model = new StreamingOnlineKMeansModel(null, null)

  def getK: Int = k

  def setK(k: Int) {
    require(k > 0, s"Number of clusters must be positive but got ${k}")
    this.k = k
  }

  def getSeed: Long = seed

  def setSeed(seed: Long) {
    this.seed = seed
  }

  def latestModel(): StreamingOnlineKMeansModel = {
    model
  }

  def setInitialCenters(centers: Array[Vector], weights: Array[Double]): this.type = {
    require(centers.size == weights.size,
      "Number of initial centers must be equal to number of weights")
    require(centers.size == k,
      s"Number of initial centers must be ${k} but got ${centers.size}")
    require(weights.forall(_ >= 0),
      s"Weight for each inital center must be nonnegative but got [${weights.mkString(" ")}]")
    model = new StreamingOnlineKMeansModel(centers, weights)
    this
  }

  def trainOn(data: DStream[Vector]) {
    data.foreachRDD { (rdd, time) =>

      // create random centers from input data if no initial ones set
      if (model.clusterWeights == null) {
        model = new StreamingOnlineKMeansModel(
          rdd.takeSample(true, k, new XORShiftRandom(this.seed).nextInt()).map(_.toDense),
          Array.fill(k)(0.0))
      }
      model = model.update(rdd)
    }
  }

}
