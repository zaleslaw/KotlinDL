/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.experimental.batchnorm

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/**
 * This examples demonstrates the inference concept:
 *
 * Weights are loaded from .h5 file, configuration is loaded from .json file.
 *
 * Model is evaluated after loading to obtain accuracy value.
 *
 * No additional training.
 *
 * No new layers are added.
 *
 * NOTE: Model and weights are resources in api module.
 */
fun main() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {

        for (layer in it.layers) {
            if (layer is BatchNorm || layer is Dense) layer.isTrainable = false
        }
        it.layers.last().isTrainable = true

        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        println(it.kGraph)
        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile)

        var accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracy")

        it.fit(dataset = train, epochs = 2, batchSize = 100)
        println(it.kGraph)
        accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println(it.kGraph)
        println("Accuracy after training $accuracy")
    }
}




