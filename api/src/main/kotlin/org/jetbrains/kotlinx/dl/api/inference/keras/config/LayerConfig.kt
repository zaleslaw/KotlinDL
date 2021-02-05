/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

internal data class LayerConfig(
    val activation: String? = null,
    val activity_regularizer: ActivityRegularizer? = null,
    val axis: Any? = null,
    var batch_input_shape: List<Int?>? = null,
    val beta_constraint: Any? = null,
    val beta_initializer: KerasInitializer? = null,
    val beta_regularizer: KerasRegularizer? = null,
    val bias_constraint: Any? = null,
    val bias_initializer: KerasInitializer? = null,
    val bias_regularizer: KerasRegularizer? = null,
    val center: Boolean? = null,
    val data_format: String? = null,
    val depth_multiplier: Int? = null,
    val depthwise_constraint: Any? = null,
    val depthwise_initializer: KerasInitializer? = null,
    val depthwise_regularizer: KerasRegularizer? = null,
    val dilation_rate: List<Int>? = null,
    val dropout: Double? = null,
    val dtype: String? = null,
    val epsilon: Double? = null,
    val filters: Int? = null,
    val gamma_constraint: Any? = null,
    val gamma_initializer: KerasInitializer? = null,
    val gamma_regularizer: KerasRegularizer? = null,
    val go_backwards: Boolean? = null,
    val implementation: Int? = null,
    val kernel_constraint: Any? = null,
    val kernel_initializer: KerasInitializer? = null,
    val kernel_regularizer: KerasRegularizer? = null,
    val kernel_size: List<Int>? = null,
    val max_value: Double? = null,
    val momentum: Double? = null,
    val moving_mean_initializer: KerasInitializer? = null,
    val moving_variance_initializer: KerasInitializer? = null,
    val name: String? = null,
    val negative_slope: Double? = null,
    val padding: KerasPadding? = null,
    val noise_shape: List<Any>? = null,
    val offset: Double? = null,
    val ragged: Boolean? = null,
    val rate: Double? = null,
    val recurrent_activation: String? = null,
    val recurrent_constraint: Any? = null,
    val recurrent_dropout: Double? = null,
    val recurrent_initializer: KerasInitializer? = null,
    val recurrent_regularizer: Any? = null,
    val return_sequences: Boolean? = null,
    val return_state: Boolean? = null,
    val pool_size: List<Int>? = null,
    val seed: Any? = null,
    val scale: Any? = null,
    val sparse: Boolean? = null,
    val stateful: Boolean? = null,
    val strides: List<Int>? = null,
    val target_shape: List<Int>? = null,
    val time_major: Boolean? = null,
    val trainable: Boolean? = true,
    val threshold: Double? = null,
    val unit_forget_bias: Boolean? = null,
    val units: Int? = null,
    val unroll: Boolean? = null,
    val use_bias: Boolean? = null
)
