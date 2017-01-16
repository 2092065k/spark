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
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


class Silhouette(
    private var model: KMeansModel = null,
    private var data: RDD[Vector] = null) extends Serializable with Logging {

  def getModel: KMeansModel = model

  def setModel(model: KMeansModel) {
    this.model = model
  }

  def getData: RDD[Vector] = data

  def setData(data: RDD[Vector]) {
    this.data = data
  }

  def runAnalysis(model: KMeansModel = model, data: RDD[Vector] = data):
    Array[(Int, Int, Double)] = {

    require(model != null, "The Model has not been set")
    require(data != null, "The input points data has not been set")

    val centerNorms = model.clusterCenters.map(Vectors.norm(_, 2.0))
    val processedCenters = model.clusterCenters.zip(centerNorms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val pointNorms = data.map(Vectors.norm(_, 2.0))
    val processedData = data.zip(pointNorms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val closestCenterPairs = processedCenters.map(center =>
      getClosestCenterPair(center, processedCenters)
    ).toMap

    val closestCenterData = processedData.map( point => KMeans.findClosest(processedCenters, point))
    val pointsWithCenterInfo = processedData.zip(closestCenterData)

    val collectedPointInfo = pointsWithCenterInfo.collect()

    val silhouettes = pointsWithCenterInfo.map{ pointInfo =>

      val point = pointInfo._1
      val closestCenterIndex = pointInfo._2._1
      val neighbouringCenterIndex = closestCenterPairs(closestCenterIndex)

      val pointsInLocalCluster = collectedPointInfo.filter(elem =>
        elem._2._1 == closestCenterIndex).map(elem => elem._1)

      val pointsInNeighbouringCluster = collectedPointInfo.filter(elem =>
        elem._2._1 == neighbouringCenterIndex).map(elem => elem._1)

      // dissimilarity to the cluster the point belongs to - a(i)
      val localDissimilarity = clusterDissimilarity(pointsInLocalCluster, point)

      // dissimilarity to the neighbouring cluster of the point - b(i)
      val neighbouringDissimilarity = clusterDissimilarity(pointsInNeighbouringCluster, point)

      // silhouette of the point - s(i)
      val silhouette = (neighbouringDissimilarity - localDissimilarity) /
        Math.max(localDissimilarity, neighbouringDissimilarity)

      (closestCenterIndex, silhouette)
    }.groupByKey().map(elem => (elem._1, elem._2.size, elem._2.sum/elem._2.size.toDouble))


    silhouettes.collect()

  }

  private def clusterDissimilarity(
      clusterPoints: Array[VectorWithNorm],
      point: VectorWithNorm): Double = {
    val squaredDistances = clusterPoints.map(elem => KMeans.fastSquaredDistance(elem, point))
    val distances = squaredDistances.map(elem => Math.sqrt(elem))
    distances.sum/distances.length
  }

  private def getClosestCenterPair(
      center: VectorWithNorm,
      allCenters: Array[VectorWithNorm]): (Int, Int) = {

    val consideredCenters = allCenters.filter(elem => elem != center)
    val closestCenterInfo = KMeans.findClosest(consideredCenters, center)
    val closestCenter = consideredCenters(closestCenterInfo._1)
    val closestCenterRealIndex = allCenters.indexOf(closestCenter)
    val consideredCenterIndex = allCenters.indexOf(center)
    (consideredCenterIndex, closestCenterRealIndex)
  }

}
