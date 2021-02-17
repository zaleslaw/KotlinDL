/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.jvm

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.history.*
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.inference.keras.loadModelLayers
import org.tensorflow.*
import java.io.File
import java.io.FileNotFoundException

/**
 * Sequential model groups a linear stack of layers into a TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @property [inputLayer] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Sequential group with [inputLayer] and [layers].
 */
public class JVMSequential(input: InputJVM, vararg layers: LayerJVM) : AutoCloseable {
    /** Logger for Sequential model. */
    public val logger: KLogger = KotlinLogging.logger {}

    /** Input layer. */
    public val inputLayer: InputJVM = input

    /** The bunch of layers. */
    public val layers: List<LayerJVM> = listOf(*layers)

    /** Layers indexed by name. */
    private var layersByName: Map<String, Layer> = mapOf()

    public companion object {
        /**
         * Creates the [JVMSequential] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [JVMSequential] model.
         */
        @JvmStatic
        public fun of(input: InputJVM, vararg layers: LayerJVM): JVMSequential {
            val seqModel = JVMSequential(input, *layers)
            return seqModel
        }

        /**
         * Creates the [JVMSequential] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [JVMSequential] model.
         */
        @JvmStatic
        public fun of(layers: List<LayerJVM>): JVMSequential {
            val input = layers[0]
            require(input is InputJVM) { "Model should start from the Input layer" }

            val otherLayers = layers.subList(1, layers.size)
            val seqModel = JVMSequential(input, *otherLayers.toTypedArray())
            return seqModel
        }

        /**
         * Creates the [JVMSequential] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [JVMSequential] model.
         */
        @JvmStatic
        public fun of(input: InputJVM, layers: List<LayerJVM>): JVMSequential {
            val seqModel = JVMSequential(input, *layers.toTypedArray())
            return seqModel
        }

        /**
         * Loads a [JVMSequential] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [JVMSequential] model.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File): JVMSequential {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return org.jetbrains.kotlinx.dl.jvm.loadModelConfiguration(configuration)
        }

        /**
         * Loads a [JVMSequential] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(modelDirectory: File): Pair<Input, MutableList<Layer>> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadModelLayers(configuration)
        }
    }

    fun predict(inputData: FloatArray): Any {
        var out: Any = inputData
        for (layer in layers) {
            out = layer.forward(out)
        }
        return out
    }

    fun loadWeights(modelDirectory: File, loadOptimizerState: Boolean) {


        // Load variables names
        val file = File("${modelDirectory.absolutePath}/variableNames.txt")

        if (!file.exists()) throw FileNotFoundException(
            "File 'variableNames.txt' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        val variableNames = file.readLines()

        if (variableNames.isNotEmpty()) {
            for (variableName in variableNames) {
                loadVariable(variableName, modelDirectory.absolutePath) // TODO: write our own custom variable loading
            }
        }
    }


    /**
     * Return layer by [layerName].
     *
     * @param [layerName] Should be existing layer name. Throws an error otherwise.
     */
    public infix fun getLayer(layerName: String): Layer {
        return layersByName[layerName] ?: error("No such layer $layerName in the model.")
    }

    override fun close() {

    }
}
