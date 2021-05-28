/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.reshape3DTo1D
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.reshapeInput

public class SwapAxis(public var axisOne: Int, public var axisTwo: Int) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        val shape = longArrayOf(inputShape.width!!.toLong(), inputShape.height!!.toLong(), inputShape.channels)
        val tensor3D = reshapeInput(data, shape)

        val tmp = shape[axisOne]
        shape[axisOne] = shape[axisTwo]
        shape[axisTwo] = tmp

        val reshaped = Array(
                shape[0].toInt()
        ) { Array(shape[1].toInt()) { FloatArray(shape[2].toInt()) } }

        for (i in 0 until shape[0].toInt()) {
            for (j in 0 until shape[1].toInt()) {
                for (k in 0 until shape[2].toInt()) {
                    reshaped[i][j][k] = tensor3D[k][j][i] // TODO: ugly, should handle correctly shapes
                }
            }
        }

        return reshape3DTo1D(reshaped, data.size)
    }
}
