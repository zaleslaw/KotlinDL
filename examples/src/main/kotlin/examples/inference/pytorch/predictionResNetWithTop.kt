/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.pytorch

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.InputType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelZoo
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.preprocessInput
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File

private const val PATH_TO_MODEL = "examples/src/main/resources/models/pytorch/resnet50jit.pt"

fun main() {
    val modelZoo = ModelZoo(commonModelDirectory = File("cache/pretrainedModels"), modelType = ModelType.ResNet_50)
    val model = modelZoo.loadModel() as Functional

    val imageNetClassLabels = modelZoo.loadClassLabels()

    val pytorchModel = Module.load(PATH_TO_MODEL)

    for (i in 1..8) {
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
            }
            transformTensor {
                swapaxis {
                    axisOne = 0
                    axisTwo = 2
                }
            }
        }

        val inputData = preprocessInput(preprocessing().first, model.inputDimensions, InputType.TORCH)

        val floatBuffer = Tensor.allocateFloatBuffer(inputData.size).put(inputData)
        val tensor = Tensor.fromBlob(floatBuffer, longArrayOf(1, 3, 224, 224))
        val iValue = IValue.from(tensor)
        val resultTensor = pytorchModel.forward(iValue).toTensor()
        println(resultTensor.shape().contentToString())
        val result = resultTensor.dataAsFloatArray
        val maxIdx = result.indexOfFirst { value -> value == result.maxOrNull()!! }
        println("Predicted object for image$i.jpg is ${imageNetClassLabels[maxIdx]}")
    }


    pytorchModel.destroy()
    model.close()
}



