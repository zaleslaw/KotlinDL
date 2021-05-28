/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.pytorch

import org.pytorch.Module

private const val PATH_TO_MODEL_1 = "examples/src/main/resources/models/onnx/mnist-8.onnx"
private const val PATH_TO_MODEL_2 = "examples/src/main/resources/models/pytorch/resnet50.onnx"
private const val PATH_TO_MODEL_3 = "examples/src/main/resources/models/onnx/resnet50notop.onnx"

fun main() {
    System.setProperty("java.library.path", "C:\\zaleslaw\\libtorch2\\lib")
    println(System.getProperty("java.library.path"))
    System.loadLibrary("pytorch_jni")
    Thread.sleep(3000)


    val model = Module.load(PATH_TO_MODEL_2)

    println(model)

    model.destroy()
}
