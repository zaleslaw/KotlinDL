/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.experimental.kotlininference

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.jvm.JVMSequential
import java.io.File

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 500
private const val PATH_TO_MODEL = "savedmodels/densemnist"

/**
 * This is a simple model based on Dense layers only.
 */
private val model = Sequential.of(
    Input(784),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeNormal(SEED)),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeNormal(SEED)),
    Dense(1024, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeNormal(SEED)),
    Dense(128, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeNormal(SEED)),
    Dense(10, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = HeNormal(SEED))
)

fun main() {
    /* val (train, test) = Dataset.createTrainAndTestDatasets(
         TRAIN_IMAGES_ARCHIVE,
         TRAIN_LABELS_ARCHIVE,
         TEST_IMAGES_ARCHIVE,
         TEST_LABELS_ARCHIVE,
         NUMBER_OF_CLASSES,
         ::extractImages,
         ::extractLabels
     )

     model.use {
         it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

         it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

         it.save(
             File(PATH_TO_MODEL),
             savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
             writingMode = WritingMode.OVERRIDE
         )

         val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

         println("Accuracy: $accuracy")
     }*/

    val model2 = JVMSequential.loadModelConfiguration(File("${PATH_TO_MODEL}/modelConfig.json"))

    model2.use {
        // Freeze conv2d layers, keep dense layers trainable
        println(it.layers.size)

        it.loadWeights(File(PATH_TO_MODEL))

        /* val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

         println("Accuracy before training $accuracyBefore")

         it.fit(
             dataset = train,
             validationRate = 0.1,
             epochs = 5,
             trainBatchSize = 1000,
             validationBatchSize = 100
         )

         val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

         println("Accuracy after training $accuracyAfterTraining")*/
    }
}
