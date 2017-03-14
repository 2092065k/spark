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

/**
 * Online K-means clustering with periodic updates to the state of the model,
 * initialized by batches of data produced by a DStream.
 */
class StreamingOnlineKMeansModel(
    override val clusterCenters: Array[Vector],
    val clusterWeights: Array[Double]) extends KMeansModel(clusterCenters) with Logging {

  /**
   * Update the state of the model in response to the latest batch of data.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   */
  def update(data: RDD[Vector]): StreamingOnlineKMeansModel = {

    val sc = data.sparkContext
    val closest = data.map(point => (this.predict(point), point))
    val dims = clusterCenters(0).size
    val bcCenters = sc.broadcast(clusterCenters)
    val bcWeights = sc.broadcast(clusterWeights)

    // partition the observations based on their proximity to an existing centroid
    val final_centers_and_weights = closest.groupByKey().mapPartitions{ pointsInfo =>
      val localCenters = KMeans.deepCopyVectorArray(bcCenters.value)
      val localWeights = bcWeights.value.clone()

      val centersValue = Array.fill(localCenters.length)(Vectors.zeros(dims))
      val weights = Array.fill(localWeights.length)(0.0)

      pointsInfo.foreach{ pointsPerCenterInfo =>
        val centerIndex = pointsPerCenterInfo._1
        val points = pointsPerCenterInfo._2
        weights(centerIndex) = localWeights(centerIndex)

        points.foreach{ point =>

          val center = localCenters(centerIndex)
          weights(centerIndex) += 1

          // add the contribution of one point
          val contrib = new DenseVector(Array.fill(dims)(0.0))
          axpy(1.0, point, contrib)
          axpy(-1.0, center, contrib)
          scal(1.0/weights(centerIndex), contrib)
          axpy(1.0, contrib, center)
          centersValue(centerIndex) = center
        }

      }
      weights.indices.filter(weights(_) > 0).map(j => (j, (centersValue(j), weights(j)))).iterator
    }.collect()

    // update the value of centroids that were influenced by the latest observations
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

/**
 * StreamingOnlineKMeans provides a means of training a cluster model
 * by periodically receiving batches of data from a DStream. Methods for
 * setting the initial state of the model and inspecting its present values
 * are also available.
 */
class StreamingOnlineKMeans (
    var k: Int,
    var seed: Long)  extends Logging with Serializable {

  protected var model = new StreamingOnlineKMeansModel(null, null)

  /**
   * Return the number of clusters.
   */
  def getK: Int = k

  /**
   * Set the number of clusters.
   */
  def setK(k: Int) {
    require(k > 0, s"Number of clusters must be positive but got ${k}")
    this.k = k
  }

  /**
   * Return the value of the seed used for random centroid initialization.
   */
  def getSeed: Long = seed

  /**
   * Set the value of the seed uesd for random centroid initialization.
   */
  def setSeed(seed: Long) {
    this.seed = seed
  }

  /**
   * Return the latest state of the model.
   */
  def latestModel(): StreamingOnlineKMeansModel = {
    model
  }

  /**
   * Manually initialize the K centroids by providing explicit values.
   *
   * @param centers The starting coordinates of the K centroids.
   * @param weights The starting weights of the K centroids.
   */
  def setInitialCenters(centers: Array[Vector], weights: Array[Double]): this.type = {
    require(centers.size == weights.size,
      "Number of initial centers must be equal to number of weights")
    require(centers.size == k,
      s"Number of initial centers must be ${k} but got ${centers.size}")
    require(weights.forall(_ >= 0),
      s"Weight for each initial center must be non-negative but got [${weights.mkString(" ")}]")
    model = new StreamingOnlineKMeansModel(centers, weights)
    this
  }

  /**
   * Initialize the K centroids randomly by sampling the first RDD from the DStream.
   */
  private def initialiseRandomModel(rdd: RDD[Vector]) {
    model = new StreamingOnlineKMeansModel(
      rdd.takeSample(true, k, new XORShiftRandom(this.seed).nextInt()).map(_.toDense),
      Array.fill(k)(0.0))
  }

  /**
   * Update the clustering model by training it on batches of data from a DStream.
   * Before data can be used for training, the method checks if the centroids
   * have been initialized. If this is not the case, the first RDD from the DStream
   * is randomly sampled to provide K initial coordinates.
   *
   * @param data DStream containing vector data
   */
  def trainOn(data: DStream[Vector]) {
    data.foreachRDD { (rdd, time) =>

      // discard empty RDDs
      if (!rdd.isEmpty()) {

        // create random centers from input data if no initial ones set
        if (model.clusterCenters == null) {
          initialiseRandomModel(rdd)
        }
        model = model.update(rdd)
      }
    }
  }

}
