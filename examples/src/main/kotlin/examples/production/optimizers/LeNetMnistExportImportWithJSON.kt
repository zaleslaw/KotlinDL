package examples.production.optimizers

import api.ModelWritingMode
import api.keras.ModelFormat
import api.keras.Sequential
import api.keras.dataset.Dataset
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import datasets.*
import examples.production.lenet5

private const val PATH_TO_MODEL = "savedmodels/lenet5KerasWithOptimizers"
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000

//0.9662998914718628 0.9695000648498535
/*
 */
fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val (newTrain, validation) = train.split(0.95)

    lenet5.use {
        it.compile(optimizer = Adam(), loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS)

        it.summary()

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE,
            verbose = true
        )

        it.save(
            pathToModelDirectory = PATH_TO_MODEL,
            saveOptimizerState = true,
            modelFormat = ModelFormat.KERAS_CONFIG_CUSTOM_VARIABLES,
            modelWritingMode = ModelWritingMode.OVERRIDE
        )

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")
    }

    val model = Sequential.load(PATH_TO_MODEL)

    model.use {
        /* // Freeze conv2d layers, keep dense layers trainable
         for (layer in it.layers) {
             if (layer::class == Conv2D::class)
                 layer.isTrainable = false
         }*/ // Didn't find variables from frozen variables

        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadVariablesFromTxtFiles(PATH_TO_MODEL, loadOptimizerState = true)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 1,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false, // for transfer learning
            isOptimizerInitRequired = false // for optimizer transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training with restored optimizer: $accuracyAfterTraining")
    }

    val model2 = Sequential.load(PATH_TO_MODEL)

    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadVariablesFromTxtFiles(PATH_TO_MODEL, loadOptimizerState = false)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 1,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = true,
            isWeightsInitRequired = false, // for transfer learning
            isOptimizerInitRequired = true // for optimizer transfer learning
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training with new optimizer: $accuracyAfterTraining")
    }
}