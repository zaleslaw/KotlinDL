/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.pytorch

import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor


/**
 * Inference model built on SavedModelBundle format to predict on images.
 */
public open class PyTorchModel : AutoCloseable {

    private lateinit var session: Module

    /** Data shape for prediction. */
    public lateinit var shape: LongArray
        private set

    public companion object {
        /**
         * Loads model from SavedModelBundle format.
         */
        public fun load(pathToModel: String): PyTorchModel {
            val model = PyTorchModel()
            model.session = Module.load(pathToModel)

            return model
        }
    }

    /**
     * Chain-like setter to set up input shape.
     *
     * @param dims The input shape.
     */
    public fun reshape(vararg dims: Long) {
        this.shape = TensorShape(1, *dims).dims()
    }

    public fun predict(inputData: FloatArray): Int {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val iValue = prepareData(inputData)
        val resultTensor = session.forward(iValue).toTensor()

        val result = resultTensor.dataAsFloatArray
        val maxIdx = result.indexOfFirst { value -> value == result.maxOrNull()!! }
        // TODO: pred functions usage
        return maxIdx
    }

    private fun prepareData(inputData: FloatArray): IValue? {
        val floatBuffer = Tensor.allocateFloatBuffer(inputData.size).put(inputData)
        val tensor = Tensor.fromBlob(floatBuffer, shape)
        val iValue = IValue.from(tensor)
        return iValue
    }

    public fun rawPredict(inputData: FloatArray): FloatArray {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }

        val iValue = prepareData(inputData)
        val resultTensor = session.forward(iValue).toTensor()
        //println(resultTensor.shape().contentToString())
        return resultTensor.dataAsFloatArray
    }


    /**
     * Find the maximum probability and return it's index.
     *
     * @param probabilities The probabilites.
     * @return The index of the max.
     */
    private fun pred(probabilities: FloatArray): Int {
        var maxVal = Float.NEGATIVE_INFINITY
        var idx = 0
        for (i in probabilities.indices) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i]
                idx = i
            }
        }
        return idx
    }

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @return Predicted class index.
     */
    public fun predict(inputData: FloatArray, inputTensorName: String, outputTensorName: String): Int {
        return 0
    }

    /**
     * Predicts labels for all [images].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [dataset] Dataset.
     */
    public fun predictAll(dataset: Dataset): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i))
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    /**
     * Predicts labels for all [images].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     *
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @param [dataset] Dataset.
     */
    public fun predictAll(dataset: Dataset, inputTensorName: String, outputTensorName: String): List<Int> {
        val predictedLabels: MutableList<Int> = mutableListOf()

        for (i in 0 until dataset.xSize()) {
            val predictedLabel = predict(dataset.getX(i), inputTensorName, outputTensorName)
            predictedLabels.add(predictedLabel)
        }

        return predictedLabels
    }

    /**
     * Evaluates [dataset] via [metric].
     *
     * NOTE: Slow method, executed on client side, not in TensorFlow.
     */
    public fun evaluate(
        dataset: Dataset,
        metric: Metrics
    ): Double {
        return if (metric == Metrics.ACCURACY) {
            var counter = 0
            for (i in 0 until dataset.xSize()) {
                val predictedLabel = predict(dataset.getX(i))
                if (predictedLabel == dataset.getY(i).toInt())
                    counter++
            }

            (counter.toDouble() / dataset.xSize())
        } else {
            Double.NaN
        }
    }


    override fun close() {
        session.destroy()
    }
}
