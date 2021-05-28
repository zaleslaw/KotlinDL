/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/**
 * The whole tensor preprocessing pipeline DSL.
 *
 * It supports the following ops:
 * - [rescaling] See [Rescaling] preprocessor.
 * - [sharpen] See [Sharpen] preprocessor.
 *
 * It's a part of the [org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing] pipeline DSL.
 */
public class TensorPreprocessing {
    /** */
    public lateinit var rescaling: Rescaling

    /** */
    public lateinit var sharpen: Sharpen

    /** */
    public lateinit var onnx: ONNXModelPreprocessor

    /** */
    public lateinit var swapaxis: SwapAxis

    /** */
    public lateinit var pyTorch: PyTorchModelPreprocessor

    /** True, if [rescaling] is initialized. */
    public val isRescalingInitialized: Boolean
        get() = ::rescaling.isInitialized

    /** True, if [sharpen] is initialized. */
    public val isSharpenInitialized: Boolean
        get() = ::sharpen.isInitialized


    /** True, if [onnx] is initialized. */
    public val isOnnxInitialized: Boolean
        get() = ::onnx.isInitialized

    /** True, if [swapaxis] is initialized. */
    public val isSwapAxisInitialized: Boolean
        get() = ::swapaxis.isInitialized

    /** True, if [pyTorch] is initialized. */
    public val isPyTorchInitialized: Boolean
        get() = ::pyTorch.isInitialized
}

/** */
public fun TensorPreprocessing.rescale(block: Rescaling.() -> Unit) {
    rescaling = Rescaling().apply(block)
}

/** */
public fun TensorPreprocessing.sharpen(block: Sharpen.() -> Unit) {
    sharpen = Sharpen().apply(block)
}

/** */
public fun TensorPreprocessing.onnx(block: ONNXModelPreprocessor.() -> Unit) {
    onnx = ONNXModelPreprocessor(null).apply(block)
}

/** */
public fun TensorPreprocessing.pytorch(block: PyTorchModelPreprocessor.() -> Unit) {
    pyTorch = PyTorchModelPreprocessor(null).apply(block)
}

/** */
public fun TensorPreprocessing.swapaxis(block: SwapAxis.() -> Unit) {
    swapaxis = SwapAxis(0,2).apply(block)
}





