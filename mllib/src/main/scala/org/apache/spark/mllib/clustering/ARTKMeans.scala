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

class ARTKMeans private(private var a: Double)  extends Serializable with Logging{

  var clusterCenters: Array[VectorWithNorm] = Array()
  var clusterWights: Array[Double] = Array()

  def getA: Double = a

  def setA(a: Double) {
    require(a > 0.0 && a <= 1.0, s"Vigilance percentage needs to be between 0 and 1 but got ${a}")
    this.a = a
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

  private def normalizeVector(x: Vector, max: Vector, min: Vector): Vector = {

    val size = x.size
    var z = Array[Double]()
    for(a <- 0 until size) {
      val elem = (x(a) - min(a))/(max(a) - min(a))
      z = z :+ elem
    }
    new DenseVector(z)
  }

  private def run(data: RDD[Vector]): KMeansModel = {

    val max = data.reduce{ case(x, y) => getLargerParts(x, y) }
    val min = data.reduce{ case(x, y) => getSmallestParts(x, y) }

    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()

    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    // ensure one center exists
    if(clusterCenters.length == 0) {
      clusterCenters = clusterCenters :+ zippedData.take(1).head
      clusterWights = clusterWights :+ 0.0
    }

    runAlgorithm(zippedData, max, min)
    norms.unpersist()

    new KMeansModel(clusterCenters.map(centerWithNorm => centerWithNorm.vector))
  }

  private def runAlgorithm(data: RDD[VectorWithNorm], max: Vector, min: Vector) {

    val sc = data.sparkContext
    val dims = data.take(1).head.vector.size
    val vigilance = a * dims

    data.collect().foreach {point =>

      val (bestCenter, cost) = KMeans.findClosest(clusterCenters, point)

      val closestCenter = clusterCenters(bestCenter).vector
      val normalizedPoint = normalizeVector(point.vector, max, min)
      val normalizedCenter = normalizeVector(closestCenter, max, min)

      val diff = (normalizedCenter.toArray, normalizedPoint.toArray).zipped.map(_ - _)
      val distArray = diff.map(elem => Math.pow(elem, 2))
      val dist = distArray.sum

      if(dist < vigilance) {

        // adjust existing center
        clusterWights(bestCenter) += 1
        val contrib = new DenseVector(Array.fill(dims)(0.0))
        axpy(1.0, point.vector, contrib)
        axpy(-1.0, closestCenter, contrib)
        scal(1.0/clusterWights(bestCenter), contrib)
        axpy(1.0, contrib, closestCenter)

      }
      else {
        clusterCenters = clusterCenters :+ point
        clusterWights = clusterWights :+ 1.0
      }
    }
  }

}

object ARTKMeans {
  def train(
     data: RDD[Vector],
     a: Double): KMeansModel = {
    new ARTKMeans(a).run(data)
  }
}
