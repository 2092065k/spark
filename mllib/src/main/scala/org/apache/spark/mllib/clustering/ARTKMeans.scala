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
  protected var model = new KMeansModel(clusterCenters.map(_.vector))

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
      s"Weight for each initial center must be non-negative but got [${weights.mkString(" ")}]")
    clusterCenters = centers.map(new VectorWithNorm(_))
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

    // edge case where all the data is converged at the same point
    if (max.equals(min)) {return new VectorWithNorm(x)}

    val size = x.size
    var z = Array[Double]()
    for(a <- 0 until size) {
      val elem = (x(a) - min(a))/(max(a) - min(a))
      z = z :+ elem
    }
    new VectorWithNorm(z)
  }

  private def combineCenters(max: Vector, min: Vector, a: Double) {

    var newClusterCenters: Array[VectorWithNorm] = Array()
    var newClusterWeights: Array[Double] = Array()
    val dims = clusterCenters(0).vector.size
    val vigilance = a * dims

    var centersWithIndices = clusterCenters.zipWithIndex
    var centerWeightsWithIndices = clusterWeights.zipWithIndex

    while (centersWithIndices.nonEmpty) {

      // take one of the old centers and find all that are within vigilance distance
      val normalizedPoint = normalizeVector(centersWithIndices(0)._1.vector, max, min)
      val normalizedCentersToMerge = centersWithIndices.map{case (point, index) =>
        (normalizeVector(point.vector, max, min), index)
      }

      val withinVigilanceCenters = normalizedCentersToMerge.filter{case (point, index) =>
        KMeans.fastSquaredDistance(point, normalizedPoint) < vigilance
      }

      val centersToMergeIndices = withinVigilanceCenters.map(elem => elem._2)

      val centersToMerge = centersWithIndices.filter{case (point, index) =>
        centersToMergeIndices.contains(index)
      }.map(elem => elem._1)

      val centerWeightsToMerge = centerWeightsWithIndices.filter{case (weight, index) =>
        centersToMergeIndices.contains(index)
      }.map(elem => elem._1)

      val centersWithWeightsToMerge = centersToMerge.zip(centerWeightsToMerge)

      val mergedPointInfo = centersWithWeightsToMerge.reduce{ (elem1, elem2) =>
        val count = elem1._2 + elem2._2
        if (count != 0) {
          scal(elem1._2 / count, elem1._1.vector)
          scal(elem2._2 / count, elem2._1.vector)
          axpy(1.0, elem2._1.vector, elem1._1.vector)
        }
        (new VectorWithNorm(elem1._1.vector), count)
      }

      newClusterCenters = newClusterCenters :+ mergedPointInfo._1
      newClusterWeights = newClusterWeights :+ mergedPointInfo._2

      centersWithIndices = centersWithIndices.filter(elem =>
        !centersToMergeIndices.contains(elem._2))

      centerWeightsWithIndices = centerWeightsWithIndices.filter(elem =>
        !centersToMergeIndices.contains(elem._2))

    }

    clusterCenters = newClusterCenters
    clusterWeights = newClusterWeights

  }

  private def update(data: RDD[Vector]): KMeansModel = {

    // recompute new min and max
    val localMax = data.reduce{ case(x, y) => getLargerParts(x, y) }
    val localMin = data.reduce{ case(x, y) => getSmallestParts(x, y) }

    if(globalMax == null || globalMin == null) {
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
    val dims = clusterCenters(0).vector.size
    val vigilance = a * dims

    val closestCenterIndices = data.map(point => KMeans.findClosest(clusterCenters, point)._1)
    val closestPairs = closestCenterIndices.zip(data)

    val final_centers_and_weights = closestPairs.groupByKey().mapPartitions {pointsInfo =>

      var localCenters = KMeans.deepCopyVectorWithNormArray(bcCenters.value)
      var localWeights = bcWeights.value.clone()

      var centersValue = Array.fill(localCenters.length)(Vectors.zeros(dims))
      var weights = Array.fill(localWeights.length)(0.0)

      pointsInfo.foreach{ pointsPerCenterInfo =>
        var centerIndex = pointsPerCenterInfo._1
        val points = pointsPerCenterInfo._2
        weights(centerIndex) = localWeights(centerIndex)

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

            weights(centerIndex) += 1
            val contrib = new DenseVector(Array.fill(dims)(0.0))
            axpy(1.0, point.vector, contrib)
            axpy(-1.0, center, contrib)
            scal(1.0/weights(centerIndex), contrib)
            axpy(1.0, contrib, center)
            val centerResult = new VectorWithNorm(center)
            localCenters(centerIndex) = centerResult
            centersValue(centerIndex) = centerResult.vector
          }
          else {
            localCenters = localCenters :+ point
            localWeights = localWeights :+ 1.0
            centersValue = centersValue :+ point.vector
            weights = weights :+ 1.0
          }
        }
      }
      weights.indices.filter(weights(_) > 0).map(j => (j, (centersValue(j), weights(j)))).iterator
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

      // discard empty RDDs
      if (!rdd.isEmpty()) {
        model = update(rdd)
      }
    }
  }

}
