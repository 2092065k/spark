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

/**
 * Silhouette is a method of interpretation and validation of consistency
 * within clusters of data. It offers a clear and simple way of representing
 * how well each data point lies within its cluster - a value in the range [-1, 1]
 * where 1 indicates a perfect cluster assignment, 0 is indifference and -1 indicating
 * a completely wrong assignment.
 *
 * @param model the cluster model which will be evaluated
 * @param data the date, whose Silhouette value will be measured
 */
class Silhouette(
    private var model: KMeansModel = null,
    private var data: RDD[Vector] = null) extends Serializable with Logging {

  /**
   * Return the model used for evaluation.
   */
  def getModel: KMeansModel = model

  /**
   * Set a model to be evaluated.
   */
  def setModel(model: KMeansModel) {
    this.model = model
  }

  /**
   * Return the used collection of data.
   */
  def getData: RDD[Vector] = data

  /**
   * Set the collection of data, whose Silhouette value will be measured.
   */
  def setData(data: RDD[Vector]) {
    this.data = data
  }

  /**
   * Implementation of the Silhouette algorithm.
   *
   * @param model the cluster model which will be evaluated
   * @param data  the date, whose Silhouette value will be measured
   * @return a collection of tuples of the form (centroid index,
   *   number of points in the cluster, average Silhouette value of the points in the cluster)
   */
  def runAnalysis(model: KMeansModel = model, data: RDD[Vector] = data):
    Array[(Int, Int, Double)] = {

    require(model != null, "The Model has not been set")
    require(data != null, "The input points data has not been set")

    val sc = data.context

    // Compute squared norms of the centroids in the model
    val centerNorms = model.clusterCenters.map(Vectors.norm(_, 2.0))
    val processedCenters = model.clusterCenters.zip(centerNorms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    // Compute squared norms of the points in the clusters
    val pointNorms = data.map(Vectors.norm(_, 2.0))
    val processedData = data.zip(pointNorms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val closestCenterPairs = processedCenters.map(center =>
      getClosestCenterPair(center, processedCenters)
    ).toMap

    val closestCenterData = processedData.map( point => KMeans.findClosest(processedCenters, point))
    val pointsWithCenterInfo = processedData.zip(closestCenterData)

    val collectedPointInfoBroadcast = sc.broadcast(pointsWithCenterInfo.collect())

    // compute the silhouette value of each point
    val silhouettes = pointsWithCenterInfo.map{ pointInfo =>

      val point = pointInfo._1
      val closestCenterIndex = pointInfo._2._1
      val neighbouringCenterIndex = closestCenterPairs(closestCenterIndex)
      val collectedPointInfo = collectedPointInfoBroadcast.value

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

  /**
   * Computes the dissimilarity of a given point with a specified cluster.
   */
  private def clusterDissimilarity(
      clusterPoints: Array[VectorWithNorm],
      point: VectorWithNorm): Double = {
    val squaredDistances = clusterPoints.map(elem => KMeans.fastSquaredDistance(elem, point))
    val distances = squaredDistances.map(elem => Math.sqrt(elem))
    distances.sum/distances.length
  }

  /**
   * For a given centroid, identify the closest different centroid.
   *
   * @param center the centroid, whose closest neighbour is being searched for
   * @param allCenters all centroids present in the model
   * @return a tuple containing the indices of the considered centroid and its closest neighbour
   */
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
