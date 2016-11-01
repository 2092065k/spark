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

class OnlineKMeans private (
    private var k: Int,
    private var seed: Long) extends Serializable with Logging {

  def getK: Int = k

  def setK(k: Int) {
    require(k > 0, s"Number of clusters must be positive but got ${k}")
    this.k = k
  }

  def getSeed: Long = seed

  def setSeed(seed: Long) {
    this.seed = seed
  }

  private def run(data: RDD[Vector]): KMeansModel = {

    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()

    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val model = runAlgorithm(zippedData)
    norms.unpersist()

    model
  }

  private def runAlgorithm(
    data: RDD[VectorWithNorm]): KMeansModel = {

    val sc = data.sparkContext
    val centers = data.takeSample(true, k, new XORShiftRandom(this.seed).nextInt()).map(_.toDense)
    val dims = centers.head.vector.size
    val bcCenters = sc.broadcast(centers)

    val finalCenters = data.mapPartitions { points =>
      val thisCenters = bcCenters.value

      val centersValue = Array.fill(thisCenters.length)(Vectors.zeros(dims))
      val counts = Array.fill(thisCenters.length)(0.0)

      points.foreach{ point =>
        val (bestCenter, cost) = KMeans.findClosest(thisCenters, point)
        val center = centersValue(bestCenter)
        counts(bestCenter) += 1

        // add the contribution of one point
        val contrib = new DenseVector(Array.fill(dims)(0.0))
        axpy(1.0, point.vector, contrib)
        axpy(-1.0, center, contrib)
        scal(1.0/counts(bestCenter), contrib)
        axpy(1.0, contrib, center)
      }

      counts.indices.filter(counts(_) > 0).map(j => (j, (centersValue(j), counts(j)))).iterator
    }.reduceByKey { case ((centValue1, count1), (centValue2, count2)) =>
      val count = count1 + count2
      scal(count1/count, centValue1)
      scal(count2/count, centValue2)
      axpy(1.0, centValue2, centValue1)
      (centValue1, count)
    }.collectAsMap()

    new KMeansModel(finalCenters.map{case(k, v) => v._1}.toArray)

  }

}

object OnlineKMeans {

  def train(
     data: RDD[Vector],
     k: Int,
     seed: Long = 1) : KMeansModel = {
    new OnlineKMeans(k, seed).run(data)
  }

}
