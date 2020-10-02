package examples.inference.keras.demo

import LeNetClassic.SEED
import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Layer
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.MaxPool2D
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.keras.loadWeightsForFrozenLayers
import datasets.Dataset
import datasets.handlers.*

/** All weigths are loaded, the model just evaluated */
fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )


    val jsonConfigFile = getJSONConfigFile()
    val (input, otherLayers) = Sequential.loadModelLayersFromConfiguration(jsonConfigFile)

    val layers = mutableListOf<Layer>()
    for (layer in otherLayers) {
        if (layer is Conv2D || layer is MaxPool2D) {
            layer.isTrainable = false
            layers.add(layer)
        }
    }

    layers.add(Flatten("new_flatten"))
    layers.add(
        Dense(
            name = "new_dense_1",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 256,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_2",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 128,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_3",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 64,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_4",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 10,
            activation = Activations.Linear
        )
    )
    val model = Sequential.of(input, layers)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = getWeightsFile()
        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false // for transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}





