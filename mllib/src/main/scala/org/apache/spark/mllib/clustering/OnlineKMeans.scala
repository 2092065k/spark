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
import org.apache.spark.util.random.XORShiftRandom

/**
 * Online K-means clustering with random centroid initialization.
 *
 * This algorithm performs a single pass over the supplied data.
 *
 * @param k the desired number of centroids.
 * @param seed a parameter used in the random sampling of the RDD to
  *            obtain coordinates for the initial centroids.
 */
class OnlineKMeans private (
    private var k: Int,
    private var seed: Long) extends Serializable with Logging {

  /**
   * Transform the provided RDD of vectors into an RDD of
   * VectorWithNorm objects and launch the algorithm.
   */
  private def run(data: RDD[Vector]): KMeansModel = {

    // Compute squared norms and cache them
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()

    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val model = runAlgorithm(zippedData)
    norms.unpersist()

    model
  }

  /**
   * Implementation of the Online K-Means algorithm.
   *
   * @param data the supplied data to be clustered.
   * @return a KMeansModel object, holding the final locations of the K centroids
   */
  private def runAlgorithm(
    data: RDD[VectorWithNorm]): KMeansModel = {

    val sc = data.sparkContext
    val numPartitions = data.getNumPartitions
    val inactivePartitions = sc.longAccumulator
    val centers = data.takeSample(true, k, new XORShiftRandom(this.seed).nextInt()).map(_.toDense)
    val dims = centers.head.vector.size
    val bcCenters = sc.broadcast(centers)

    // partition the given data randomly and run a sequential
    // version of the algorithm for each partition
    val partialSolutionCenters = data.mapPartitions { points =>

      val localCenters = KMeans.deepCopyVectorWithNormArray(bcCenters.value)
      val weights = Array.fill(localCenters.length)(0.0)

      points.foreach{ point =>

        val (centerIndex, cost) = KMeans.findClosest(localCenters, point)
        val center = localCenters(centerIndex).vector
        weights(centerIndex) += 1

        // add the contribution of one point
        val contrib = new DenseVector(Array.fill(dims)(0.0))
        axpy(1.0, point.vector, contrib)
        axpy(-1.0, center, contrib)
        scal(1.0/weights(centerIndex), contrib)
        axpy(1.0, contrib, center)
        val centerResult = new VectorWithNorm(center)
        localCenters(centerIndex) = centerResult
      }

      if(weights.sum == 0) {
        inactivePartitions.add(1)
        Array().iterator
      }
      else {
        weights.indices.map(j => (localCenters(j), weights(j))).iterator
      }
    }.collect()

    // determine the number of partitions that operated on the given data
    val activePartitions = numPartitions - inactivePartitions.value.toInt
    var uncombinedCenters = partialSolutionCenters

    // given N active partitions that each produced K tuples as output
    // merge with a weighted average each group of N closest centroids
    val completeSolutionCenters = (0 until k).toArray.map{ centerNum =>
      val partialCenter = uncombinedCenters(0)
      val partialCenters = uncombinedCenters.sortBy(elem =>
        KMeans.fastSquaredDistance(elem._1, partialCenter._1)).take(activePartitions)

      uncombinedCenters = uncombinedCenters.filter(!partialCenters.contains(_))
      val convertedPartialCenters = partialCenters.map(elem => (elem._1.vector, elem._2))

      convertedPartialCenters.reduce { (elem1, elem2) =>
        val count = elem1._2 + elem2._2
        if (count != 0) {
          scal(elem1._2 / count, elem1._1)
          scal(elem2._2 / count, elem2._1)
          axpy(1.0, elem2._1, elem1._1)
        }
        (elem1._1, count)
      }._1
    }

    new KMeansModel(completeSolutionCenters)

  }

}

/**
 * Top level object for calling Online K-means clustering
 */
object OnlineKMeans {

  /**
   * Trains an online k-means model.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param seed Random seed for cluster initialization.
   * @return
   */
  def train(
     data: RDD[Vector],
     k: Int,
     seed: Long = 1) : KMeansModel = {
    new OnlineKMeans(k, seed).run(data)
  }

}
