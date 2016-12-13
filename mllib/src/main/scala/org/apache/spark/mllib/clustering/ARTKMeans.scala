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

class ARTKMeans (private var a: Double)  extends Logging with Serializable {

  protected var clusterCenters: Array[VectorWithNorm] = Array()
  protected var clusterWeights: Array[Double] = Array()
  protected var globalMax: Vector = null
  protected var globalMin: Vector = null
  protected var model = new KMeansModel(clusterCenters.map(centerWithNorm => centerWithNorm.vector))

  def getA: Double = a

  def setA(a: Double) {
    require(a > 0.0 && a <= 1.0, s"Vigilance percentage needs to be between 0 and 1 but got ${a}")
    this.a = a
  }

  def latestModel(): KMeansModel = {
    model
  }

  def setInitialCenters(centers: Array[Vector], weights: Array[Double]): this.type = {
    require(centers.size == weights.size,
      "Number of initial centers must be equal to number of weights")
    require(weights.forall(_ >= 0),
      s"Weight for each inital center must be nonnegative but got [${weights.mkString(" ")}]")
    clusterCenters = centers.zip(centers.map(Vectors.norm(_, 2.0))).map{ case (v, norm) =>
      new VectorWithNorm(v, norm)
    }
    clusterWeights = weights
    model = new KMeansModel(centers)
    this
  }

  private def getLargerParts(x: Vector, y: Vector): Vector = {
    val size = x.size
    var z = Array[Double]()
    for(a <- 0 until size) {
      val max = Math.max(x(a), y(a))
      z = z :+ max
    }
    new DenseVector(z)
  }

  private def getSmallestParts(x: Vector, y: Vector): Vector = {
    val size = x.size
    var z = Array[Double]()
    for(a <- 0 until size) {
      val min = Math.min(x(a), y(a))
      z = z :+ min
    }
    new DenseVector(z)
  }

  private def normalizeVector(x: Vector, max: Vector, min: Vector): VectorWithNorm = {

    val size = x.size
    var z = Array[Double]()
    for(a <- 0 until size) {
      val elem = (x(a) - min(a))/(max(a) - min(a))
      z = z :+ elem
    }
    new VectorWithNorm(z)
  }

  /**
    * Combination of the centers works through performing a sequential
    * ARTKMeans clustering on the center points themselves
    */
  private def combineCenters(max: Vector, min: Vector, a: Double) {

    var newClusterCenters: Array[VectorWithNorm] = Array()
    var newClusterWeights: Array[Double] = Array()
    val dims = clusterCenters(0).vector.size
    val vigilance = a * dims

    // take one of the old centers as an initial center
    newClusterCenters = newClusterCenters :+ clusterCenters(0)
    newClusterWeights = newClusterWeights :+ 0.0

    clusterCenters.zipWithIndex.foreach{ case(point, index) =>

      val (bestCenter, distance) = KMeans.findClosest(newClusterCenters, point)
      val closestCenter = newClusterCenters(bestCenter).vector
      val normalizedPoint = normalizeVector(point.vector, max, min)
      val normalizedCenter = normalizeVector(closestCenter, max, min)
      val dist = KMeans.fastSquaredDistance(normalizedPoint, normalizedCenter)

      if(dist < vigilance) {

        val newWeight = newClusterWeights(bestCenter) + clusterWeights(index)

        // check is necessary to overcome initial center placement
        if(newWeight != 0) {
          val contrib = new DenseVector(Array.fill(dims)(0.0))
          axpy(clusterWeights(index) / newWeight, point.vector, contrib)
          scal(newClusterWeights(bestCenter) / newWeight, closestCenter)
          axpy(1.0, contrib, closestCenter)
          newClusterWeights(bestCenter) = newWeight
        }
      }
      else {
        newClusterCenters = newClusterCenters :+ point
        newClusterWeights = newClusterWeights :+ clusterWeights(index)
      }
    }

    clusterCenters = newClusterCenters
    clusterWeights = newClusterWeights

  }

  private def update(data: RDD[Vector]): KMeansModel = {

    // recompute new min and max
    val localMax = data.reduce{ case(x, y) => getLargerParts(x, y) }
    val localMin = data.reduce{ case(x, y) => getSmallestParts(x, y) }

    if(globalMax == null || localMax == null) {
      globalMax = localMax
      globalMin = localMin
    }

    globalMax = getLargerParts(globalMax, localMax)
    globalMin = getSmallestParts(globalMin, localMin)

    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()

    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    // ensure one center exists
    if(clusterCenters.length == 0) {
      clusterCenters = clusterCenters :+ zippedData.take(1).head
      clusterWeights = clusterWeights :+ 0.0
    }

    // the state space may have expanded when reading new data
    combineCenters(globalMax, globalMin, a)

    runAlgorithm(zippedData, globalMax, globalMin)
    norms.unpersist()

    new KMeansModel(clusterCenters.map(centerWithNorm => centerWithNorm.vector))
  }

  private def runAlgorithm(data: RDD[VectorWithNorm], max: Vector, min: Vector) {

    val sc = data.sparkContext
    val bcCenters = sc.broadcast(clusterCenters)
    val bcWeights = sc.broadcast(clusterWeights)
    val initialCenterCount = clusterCenters.length
    val dims = data.take(1).head.vector.size
    val vigilance = a * dims

    val closestCenterIndices = data.map(point => KMeans.findClosest(clusterCenters, point)._1)
    val closestPairs = closestCenterIndices.zip(data)

    val final_centers_and_weights = closestPairs.groupByKey().mapPartitions {pointsInfo =>

      var localCenters = bcCenters.value
      var localWeights = bcWeights.value

      var centersValue = Array.fill(localCenters.length)(Vectors.zeros(dims))
      var counts = Array.fill(localWeights.length)(0.0)

      pointsInfo.foreach{ pointsPerCenterInfo =>
        var centerIndex = pointsPerCenterInfo._1
        val points = pointsPerCenterInfo._2
        counts(centerIndex) = localWeights(centerIndex)

        points.foreach { point =>

          /*
          need to recompute the center for each point to know if it should contribute to
          the initially closest center or to the newly added ones and get its correct index
          */
          centerIndex = KMeans.findClosest(localCenters, point)._1
          val center = localCenters(centerIndex).vector

          val normalizedPoint = normalizeVector(point.vector, max, min)
          val normalizedCenter = normalizeVector(center, max, min)

          val dist = KMeans.fastSquaredDistance(normalizedCenter, normalizedPoint)

          if(dist < vigilance) {

            counts(centerIndex) += 1
            val contrib = new DenseVector(Array.fill(dims)(0.0))
            axpy(1.0, point.vector, contrib)
            axpy(-1.0, center, contrib)
            scal(1.0/counts(centerIndex), contrib)
            axpy(1.0, contrib, center)
            centersValue(centerIndex) = center
          }
          else {
            localCenters = localCenters :+ point
            localWeights = localWeights :+ 1.0
            centersValue = centersValue :+ point.vector
            counts = counts :+ 1.0
          }
        }
      }
      counts.indices.filter(counts(_) > 0).map(j => (j, (centersValue(j), counts(j)))).iterator
    }.collect()

    final_centers_and_weights.foreach { centerInfo =>
      val centerIndex = centerInfo._1
      val newCenterValue = centerInfo._2._1
      val newCenterWeight = centerInfo._2._2

      if(centerIndex < initialCenterCount) {

        // alter the value of an existing center
        clusterWeights(centerIndex) = newCenterWeight
        clusterCenters(centerIndex) = new VectorWithNorm(newCenterValue)
      }
      else {

        // include the new center
        clusterWeights = clusterWeights :+ newCenterWeight
        clusterCenters = clusterCenters :+ new VectorWithNorm(newCenterValue)
      }

    }

    // attempt to recombine centers again to remove extras on the borders of partitions
    combineCenters(globalMax, globalMin, a/2.0)
  }

  def trainOn(data: DStream[Vector]) {
    data.foreachRDD { (rdd, time) =>
      model = update(rdd)
    }
  }

}