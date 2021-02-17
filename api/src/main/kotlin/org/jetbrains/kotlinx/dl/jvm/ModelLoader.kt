/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.jvm

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.jetbrains.kotlinx.dl.api.inference.keras.config.KerasLayer
import org.jetbrains.kotlinx.dl.api.inference.keras.config.KerasSequentialModel
import org.jetbrains.kotlinx.dl.api.inference.keras.config.LayerConfig
import java.io.File

/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadModelConfiguration(
    configuration: File
): JVMSequential {
    val pair = loadModelLayers(configuration)
    val input: InputJVM = pair.first
    val layers = pair.second

    return JVMSequential.of(input, layers.toList())
}

/**
 * Loads a [Sequential] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param jsonConfigFile File containing model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadModelLayers(jsonConfigFile: File): Pair<InputJVM, MutableList<LayerJVM>> {
    val sequentialConfig = try {
        val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
        Klaxon()
            .parse<KerasSequentialModel>(jsonString)
    } catch (e: Exception) {
        e.printStackTrace()
        try {
            Klaxon()
                .parse<KerasSequentialModel>(jsonConfigFile)
        } catch (e: Exception) {
            e.printStackTrace()
            throw IllegalArgumentException("JSON file: ${jsonConfigFile.name} contains invalid JSON. The model configuration could not be loaded from this file.")
        }
    }

    val layers = mutableListOf<LayerJVM>()

    (sequentialConfig as KerasSequentialModel).config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it, layers)
                layers.add(layer)
            }
        }
    }

    val input: InputJVM

    val batchInputShape = sequentialConfig.config!!.layers!!.first().config!!.batch_input_shape

    // TODO: write more universal code here
    val size = batchInputShape!!.size
    if (size == 2) {
        input = InputJVM(
            batchInputShape!![1]?.toLong()!!,
        )
    } else if (size == 3) {
        input = InputJVM(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!
        )
    } else if (size == 4) {
        input = InputJVM(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!,
            batchInputShape[3]?.toLong()!!
        )
    } else {
        input = InputJVM(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!,
            batchInputShape[3]?.toLong()!!
        )
    }

    return Pair(input, layers)
}

private fun convertToLayer(kerasLayer: KerasLayer, layers: MutableList<LayerJVM>): LayerJVM {
    return when (kerasLayer.class_name) {
        LAYER_DENSE -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported yet!")
    }
}


private fun createDense(config: LayerConfig, name: String): DenseJVM {
    return DenseJVM(
        outputSize = config.units!!,
        activation = convertToActivation(config.activation!!),
        name = name
    )
}

private fun convertToActivation(activation: String): Activations {
    return when (activation) {
        ACTIVATION_RELU -> Activations.Relu
        ACTIVATION_SIGMOID -> Activations.Sigmoid
        ACTIVATION_SOFTMAX -> Activations.Softmax
        ACTIVATION_LINEAR -> Activations.Linear
        ACTIVATION_TANH -> Activations.Tanh
        ACTIVATION_RELU6 -> Activations.Relu6
        ACTIVATION_ELU -> Activations.Elu
        ACTIVATION_SELU -> Activations.Selu
        ACTIVATION_LOG_SOFTMAX -> Activations.LogSoftmax
        ACTIVATION_EXP -> Activations.Exponential
        ACTIVATION_SOFTPLUS -> Activations.SoftPlus
        ACTIVATION_SOFTSIGN -> Activations.SoftSign
        ACTIVATION_HARD_SIGMOID -> Activations.HardSigmoid
        ACTIVATION_SWISH -> Activations.Swish
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}



