/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.pytorch

import org.pytorch.Module

private const val PATH_TO_MODEL_1 = "examples/src/main/resources/models/onnx/mnist-8.onnx"
private const val PATH_TO_MODEL_2 = "examples/src/main/resources/models/pytorch/resnet50jit.pt"
private const val PATH_TO_MODEL_3 = "examples/src/main/resources/models/onnx/resnet50notop.onnx"

// Ubuntu: -Djava.library.path="/home/zaleslaw/torch_versions/1.8.java/lib"
// Windows: need more additional DLL from Visual C++
fun main() {
    val model = Module.load(PATH_TO_MODEL_2)

    println(model)

    model.destroy()
}
