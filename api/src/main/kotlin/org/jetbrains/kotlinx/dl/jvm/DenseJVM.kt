/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.jvm

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import kotlin.math.exp
import kotlin.math.max

private const val KERNEL = "dense_kernel"
private const val BIAS = "dense_bias"

/**
 * Densely-connected (fully-connected) layer class.
 *
 * This layer implements the operation:
 * `outputs = activation(inputs * kernel + bias)`
 * @constructor Creates [DenseJVM] object.
 */
public class DenseJVM(
    public val outputSize: Int = 128,
    public val activation: Activations = Activations.Relu,
    name: String = ""
) : LayerJVM(name) {

    // weight tensors
    private var kernel = FloatArray(outputSize) { 0.0f }
    private var bias: Float = 0.0f


    override fun forward(
        inputData: Any
    ): Any {

        inputData as FloatArray

        val signal = inputData * kernel + bias
        return activation(signal)

    }

    private fun activation(signal: Float): Float {
        return when (activation) {
            Activations.Linear -> signal
            Activations.Sigmoid -> sigmoid(signal)
            Activations.Relu -> relu(signal)

            else -> throw UnsupportedOperationException()
        }
    }

    private operator fun FloatArray.times(kernel: FloatArray): Float {
        var sum = 0.0f
        val len = kernel.size
        for (i in 0 until len) sum += this[i] * kernel[i]
        return sum
    }

    private fun sigmoid(z: Float): Float {
        return 1.0f / (1.0f + exp(-z))
    }

    private fun relu(z: Float): Float {
        return max(0.0f, z)
    }

}
