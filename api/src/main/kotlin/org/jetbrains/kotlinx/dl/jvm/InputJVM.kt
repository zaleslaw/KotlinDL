/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.jvm

/**
 * First and required layer in [org.jetbrains.kotlinx.dl.api.core.Sequential.of] method.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [InputJVM] layer from [packedDims] representing [input] data shape.
 */
public class InputJVM(vararg dims: Long, name: String = "") : LayerJVM(name) {
    public val packedDims: LongArray = dims

    override fun forward(
        input: Any,
    ): Any {
        return input
    }
}
