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
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.random.XORShiftRandom

class SOMKMeans (
    var NNDimensions: Int,
    var nSize: Int,
    var sigma: Double,
    var learningRate: Double,
    var seed: Long)  extends Logging with Serializable {

  var clusterCenters: Array[VectorWithNorm] = null
  var model: KMeansModel = null

  def getNNDimensions: Int = NNDimensions

  def setNNDimensions(NNDimensions: Int) {
    require(NNDimensions > 0,
      s"The dimensions of the neural network must be positive but got ${NNDimensions}")
    this.NNDimensions = NNDimensions
  }

  def getNSize: Int = nSize

  def setNSize(nSize: Int) {
    require(nSize >= 0,
      s"The size of the neighbourhood must be non-negative but got ${nSize}")
    require(nSize < NNDimensions,
      s"The size of the neighbourhood of a centroid must be" +
        s" smaller than the dimensions of the centroid matrix")
    this.nSize = nSize
  }

  def getSigma: Double = sigma

  def setSigma(sigma: Double) {
    require(sigma >= 1,
      s"Sigma must be a real number grater or equal to 1 but got ${sigma}")
    this.sigma = sigma
  }

  def getLearningRate: Double = learningRate

  def setLearningRate(learningRate: Double) {
    require(learningRate >= 0 && learningRate <= 1,
      s"The learning rate must be a real number between 0 and 1 ${learningRate}")
    this.nSize = nSize
  }

  def getSeed: Long = seed

  def setSeed(seed: Long) {
    this.seed = seed
  }

  def setInitialCenters(centers: Array[Vector]): this.type = {
    require(centers.size == NNDimensions * NNDimensions,
      s"Number of initial centers must be ${NNDimensions * NNDimensions} but got ${centers.size}")
    clusterCenters = centers.zip(centers.map(Vectors.norm(_, 2.0))).map{ case (v, norm) =>
      new VectorWithNorm(v, norm)
    }
    model = new KMeansModel(centers)
    this
  }

  def latestModel(): KMeansModel = model

  private def initialiseRandomModel(rdd: RDD[Vector]) {

    val numCenters = NNDimensions * NNDimensions
    val randomCenters = rdd.takeSample(true, numCenters, new XORShiftRandom(this.seed).nextInt())
    clusterCenters = randomCenters.zip(randomCenters.map(
      Vectors.norm(_, 2.0))).map{ case (v, norm) => new VectorWithNorm(v, norm)
    }
    model = new KMeansModel(randomCenters)
  }

  private def nonNegativeMod(x: Int, y: Int): Int = {
    ((x % y) + y) % y
  }

  private def neighbourhoodImpact (x: VectorWithNorm, y: VectorWithNorm) : Double = {
    val squaredDistance = KMeans.fastSquaredDistance(x, y)
    math.exp(-sigma * squaredDistance)
  }

  private def update(data: RDD[Vector]): KMeansModel = {

    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()

    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }


    val model = runAlgorithm(zippedData)
    norms.unpersist()

    model
  }

  private def runAlgorithm(data: RDD[VectorWithNorm]): KMeansModel = {

    val sc = data.sparkContext
    val bcCenters = sc.broadcast(clusterCenters)
    val dims = clusterCenters(0).vector.size

    val final_centers_and_weights = data.mapPartitions { points =>

      val localCenters = KMeans.deepCopyVectorWithNormArray(bcCenters.value)
      val addedWeights = Array.fill(clusterCenters.length)(0.0)

      points.foreach { point =>

        val (centerIndex, cost) = KMeans.findClosest(localCenters, point)
        val center = localCenters(centerIndex)
        addedWeights(centerIndex) += 1

        // adjust the closest centroid
        val contrib = new DenseVector(Array.fill(dims)(0.0))
        axpy(1.0, point.vector, contrib)
        axpy(-1.0, center.vector, contrib)
        scal(learningRate, contrib)
        axpy(1.0, contrib, center.vector)

        // obtain the indices of all neighbour centroids
        val winnerRow = centerIndex / NNDimensions
        val winnerCol = centerIndex % NNDimensions
        val neighbourhoodCentroidIndices =
          (-nSize to nSize).flatMap { rowNumber =>
            (-nSize to nSize).map { colNumber =>
              val row = nonNegativeMod(winnerRow + rowNumber, NNDimensions)
              val col = nonNegativeMod(winnerCol + colNumber, NNDimensions)
              val neighbourIndex = row * NNDimensions + col
              neighbourIndex
            }
          }.filter(index => index != centerIndex)

        // adjust the centroids in the neighbourhood
        neighbourhoodCentroidIndices.foreach{ index =>

          val neighbourCenter = localCenters(index)
          val ni = neighbourhoodImpact(center, neighbourCenter)
          addedWeights(index) += ni

          val contrib = new DenseVector(Array.fill(dims)(0.0))
          axpy(1.0, point.vector, contrib)
          axpy(-1.0, neighbourCenter.vector, contrib)
          scal(learningRate * ni, contrib)
          axpy(1.0, contrib, neighbourCenter.vector)
        }
      }
      addedWeights.indices.filter(addedWeights(_) > 0).map(j =>
        (j, (localCenters(j), addedWeights(j)))).iterator
    }.reduceByKey{ case ((center1, addedCount1), (center2, addedCount2)) =>
      val addedCount = addedCount1 + addedCount2
      scal(addedCount1/addedCount, center1.vector)
      scal(addedCount2/addedCount, center2.vector)
      axpy(1.0, center2.vector, center1.vector)
      (center1, addedCount)
    }.collect()

    final_centers_and_weights.foreach{ centerInfo =>
      val centerIndex = centerInfo._1
      val newCenterValue = centerInfo._2._1
      clusterCenters(centerIndex) = newCenterValue
    }

    new KMeansModel(clusterCenters.map(elem => elem.vector))

  }

  def trainOn(data: DStream[Vector]) {

    data.foreachRDD { (rdd, time) =>
      // create random centers from input data if no initial ones set
      if (model == null) {
        initialiseRandomModel(rdd)
      }
      model = update(rdd)
    }
  }

}
