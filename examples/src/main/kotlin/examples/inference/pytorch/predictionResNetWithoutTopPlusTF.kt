/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.pytorch


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelZoo
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxModel
import org.jetbrains.kotlinx.dl.api.inference.pytorch.PyTorchModel
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/pytorch/resnet50withoutTopjit.pt"
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 16
private const val TEST_BATCH_SIZE = 32
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 128L
private const val TRAIN_TEST_SPLIT_RATIO = 0.8

/**
 * This is a simple model based on Dense layers only.
 */
private val topModel = Sequential.of(
        Input(4, 4, 2048),
        GlobalAvgPool2D(),
        Dense(2, Activations.Linear, kernelInitializer = HeNormal(12L), biasInitializer = Zeros())
)

fun main() {
    val modelZoo = ModelZoo(commonModelDirectory = File("cache/pretrainedModels"), modelType = ModelType.ResNet_50)
    val model = modelZoo.loadModel() as Functional

    val imageNetClassLabels = modelZoo.loadClassLabels()

    val resnet50 = PyTorchModel.load(PATH_TO_MODEL)

    resnet50.use {
        it.reshape(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        val dogsVsCatsDatasetPath = dogsCatsDatasetPath()

        val preprocessing: Preprocessing = preprocess {
            transformImage {
                load {
                    pathToData = File(dogsVsCatsDatasetPath)
                    imageShape = ImageShape(channels = NUM_CHANNELS)
                    colorMode = ColorOrder.BGR
                    labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
                }
                resize {
                    outputHeight = IMAGE_SIZE.toInt()
                    outputWidth = IMAGE_SIZE.toInt()
                    interpolation = InterpolationType.BILINEAR
                }
            }
            transformTensor {
                swapaxis {
                    axisOne = 0
                    axisTwo = 2
                }
                sharpen {
                    modelType = ModelType.ResNet_50_TORCH
                }
                pytorch {
                    pyTorchModel = resnet50
                }
            }
        }

        val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
        val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

        topModel.use {
            topModel.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            topModel.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

            val accuracy = topModel.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")
        }
    }
}


