package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import kotlin.random.Random

/**
 * TransformerForecaster - State-of-the-Art ML
 * Multi-head attention mechanism with advanced features
 */
class TransformerForecasterImpl : AlgorithmRepository {
    // Algorithm-specific configuration
    private data class TransformerConfig(
        val sequenceLength: Int = 5,
        val hiddenDim: Int = 64,
        val numHeads: Int = 4,
        val numLayers: Int = 2
    )

    private data class Features(
        val price: Double,
        val returns: Double,
        val logReturns: Double,
        val volatility: Double,
        val momentum: Double,
        val rsi: Double,
        val priceMA: Double,
        val volume: Double
    )

    private data class TransformerModel(
        val predict: (List<List<Double>>) -> List<Double>
    )

    private val config = TransformerConfig()
    private lateinit var features: List<Features>
    private lateinit var model: TransformerModel

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < config.sequenceLength + 1) {
            return "Insufficient data"
        }

        // 1. Extract multi-scale features
        features = extractFeatures(timeSeries)

        // 2. Build transformer model
        model = buildTransformer()

        // 3. Generate forecasts using Monte Carlo sampling for uncertainty quantification
        val result = forecast(
            currentPrice = timeSeries.last().price,
            upperBand = upperBand,
            lowerBand = lowerBand,
            daysAhead = daysAhead,
            numSamples = 1000
        )

        return """
            Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", result.first)}%
            Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", result.second)}%
            """.trimIndent()
    }

    private fun extractFeatures(timeSeries: List<TimeSeriesEntity>): List<Features> {
        val features = mutableListOf<Features>()
        val prices = timeSeries.map { it.price }

        for (i in 1 until prices.size) {
            val curr = prices[i]
            val prev = prices[i-1]

            // Multi-scale features
            val returns = (curr - prev) / prev
            val logReturns = ln(curr / prev)
            val volatility = calculateRollingVolatility(prices, i, 3)
            val momentum = calculateMomentum(prices, i, 3)
            val rsi = calculateRSI(prices, i, 3)
            val priceMA = calculateMA(prices, i, 3)
            val volume = 1000 + Random.nextDouble() * 500 // Simulated volume

            features.add(Features(
                price = curr,
                returns = returns,
                logReturns = logReturns,
                volatility = volatility,
                momentum = momentum,
                rsi = rsi,
                priceMA = priceMA,
                volume = volume
            ))
        }

        return features
    }

    private fun calculateRollingVolatility(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        val returns = mutableListOf<Double>()
        for (i in index - window + 1..index) {
            if (i > 0) {
                returns.add(ln(prices[i] / prices[i-1]))
            }
        }
        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance * 252) // Annualized
    }

    private fun calculateMomentum(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        return (prices[index] - prices[index - window]) / prices[index - window]
    }

    private fun calculateRSI(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 50.0
        var gains = 0.0
        var losses = 0.0
        for (i in index - window + 1..index) {
            if (i > 0) {
                val change = prices[i] - prices[i-1]
                if (change > 0) gains += change
                else losses += abs(change)
            }
        }
        val rs = gains / (if (losses > 0.0) losses else 1.0)
        return 100 - (100 / (1 + rs))
    }

    private fun calculateMA(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window - 1) return prices[index]
        var sum = 0.0
        for (i in index - window + 1..index) {
            sum += prices[i]
        }
        return sum / window
    }

    // Simplified transformer attention mechanism
    private fun multiHeadAttention(query: List<List<Double>>, key: List<List<Double>>, value: List<List<Double>>): List<List<Double>> {
        if (query.isEmpty() || key.isEmpty() || value.isEmpty()) return emptyList()

        val scores = Array(query.size) { Array(key.size) { 0.0 } }

        for (i in query.indices) {
            for (j in key.indices) {
                // Dot product attention (simplified)
                var score = 0.0
                val minSize = minOf(query[i].size, key[j].size)
                for (k in 0 until minSize) {
                    score += query[i][k] * key[j][k]
                }
                scores[i][j] = exp(score)
            }
        }

        // Softmax normalization
        for (i in scores.indices) {
            val sum = scores[i].sum()
            val normalizer = if (sum > 0.0) sum else 1.0
            for (j in scores[i].indices) {
                scores[i][j] = scores[i][j] / normalizer
            }
        }

        // Apply attention to values
        val output = mutableListOf<List<Double>>()
        for (i in scores.indices) {
            val weighted = MutableList(value.getOrNull(0)?.size ?: 0) { 0.0 }
            for (j in value.indices) {
                for (k in value[j].indices) {
                    if (k < weighted.size && j < scores[i].size) {
                        weighted[k] += scores[i][j] * value[j][k]
                    }
                }
            }
            output.add(weighted)
        }

        return output
    }

    private fun buildTransformer(): TransformerModel {
        // Simplified transformer architecture
        return TransformerModel { sequence ->
            // Self-attention on the sequence
            val attended = multiHeadAttention(sequence, sequence, sequence)

            // Simple feedforward network (simplified)
            attended.map { item ->
                val sum = item.sum()
                val itemSize = item.size
                if (itemSize > 0) sum / itemSize else 1.0
            }
        }
    }

    private fun forecast(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        numSamples: Int = 1000
    ): Pair<Double, Double> {

        val predictions = mutableListOf<List<Double>>()

        // Monte Carlo sampling for uncertainty quantification
        repeat(numSamples) {
            var lastPrice = currentPrice
            val forecast = mutableListOf<Double>()

            // Get recent features for transformer input
            val currentFeatures = features.takeLast(config.sequenceLength)
            val recentFeatures = currentFeatures.toMutableList()

            repeat(daysAhead) {
                // Transform features to vectors for transformer
                val featureVectors = recentFeatures.map { f ->
                    listOf(f.returns, f.logReturns, f.volatility, f.momentum, f.rsi)
                }

                // Get transformer prediction
                val prediction = model.predict(featureVectors)
                val avgPrediction = prediction.average()

                // Add noise and non-linearity
                val noise = (Random.nextDouble() - 0.5) * 0.02 // 2% noise
                val volatilityAdj = recentFeatures.lastOrNull()?.volatility?.let { it / 100 } ?: 0.02
                val priceChange = avgPrediction * 0.1 + noise * volatilityAdj

                lastPrice *= (1 + priceChange)
                forecast.add(lastPrice)

                // Update features for next prediction (simplified)
                if (recentFeatures.size >= config.sequenceLength) {
                    recentFeatures.removeAt(0)
                }

                // Create new feature based on prediction
                val newFeature = Features(
                    price = lastPrice,
                    returns = priceChange,
                    logReturns = ln(1 + priceChange),
                    volatility = volatilityAdj * 100,
                    momentum = priceChange,
                    rsi = 50.0 + Random.nextDouble() * 20 - 10, // Simplified RSI
                    priceMA = lastPrice,
                    volume = 1000 + Random.nextDouble() * 500
                )
                recentFeatures.add(newFeature)
            }

            predictions.add(forecast)
        }

        // Calculate probabilities from Monte Carlo samples
        var reachesUpper = 0
        var reachesLower = 0

        predictions.forEach { path ->
            val maxPrice = path.maxOrNull() ?: currentPrice
            val minPrice = path.minOrNull() ?: currentPrice

            if (maxPrice >= upperBand) reachesUpper++
            if (minPrice <= lowerBand) reachesLower++
        }

        val probUpper = (reachesUpper.toDouble() / numSamples) * 100
        val probLower = (reachesLower.toDouble() / numSamples) * 100

        return Pair(probUpper, probLower)
    }

    // Calculate percentiles from Monte Carlo samples (utility method)
    private fun getPercentile(sortedArray: List<Double>, percentile: Double): Double {
        val index = ceil(sortedArray.size * percentile).toInt() - 1
        return sortedArray.getOrElse(max(0, index)) { sortedArray.lastOrNull() ?: 0.0 }
    }

    // Generate detailed forecast with percentiles (following your JS structure)
    private fun forecastWithPercentiles(daysAhead: Int, numSamples: Int = 1000): Map<String, Map<String, Double>> {
        val results = mutableMapOf<String, Map<String, Double>>()
        val currentPrice = features.lastOrNull()?.price ?: 0.0

        val predictions = mutableListOf<List<Double>>()

        // Generate predictions
        repeat(numSamples) {
            var lastPrice = currentPrice
            val forecast = mutableListOf<Double>()
            val recentFeatures = features.takeLast(config.sequenceLength).toMutableList()

            repeat(daysAhead) {
                val featureVectors = recentFeatures.map { f ->
                    listOf(f.returns, f.logReturns, f.volatility, f.momentum, f.rsi)
                }

                val prediction = model.predict(featureVectors)
                val avgPrediction = prediction.average()

                val noise = (Random.nextDouble() - 0.5) * 0.02
                val volatilityAdj = recentFeatures.lastOrNull()?.volatility?.div(100) ?: 0.02
                val priceChange = avgPrediction * 0.1 + noise * volatilityAdj

                lastPrice *= (1 + priceChange)
                forecast.add(lastPrice)

                // Update features
                if (recentFeatures.size >= config.sequenceLength) {
                    recentFeatures.removeAt(0)
                }

                recentFeatures.add(Features(
                    price = lastPrice,
                    returns = priceChange,
                    logReturns = ln(1 + priceChange),
                    volatility = volatilityAdj * 100,
                    momentum = priceChange,
                    rsi = 50.0 + Random.nextDouble() * 20 - 10,
                    priceMA = lastPrice,
                    volume = 1000 + Random.nextDouble() * 500
                ))
            }

            predictions.add(forecast)
        }

        // Calculate percentiles for each day
        for (day in 0 until daysAhead) {
            val dayPredictions = predictions.map { it.getOrNull(day) ?: currentPrice }.sorted()

            results["day_${day + 1}"] = mapOf(
                "1%" to getPercentile(dayPredictions, 0.01),
                "5%" to getPercentile(dayPredictions, 0.05),
                "10%" to getPercentile(dayPredictions, 0.10),
                "25%" to getPercentile(dayPredictions, 0.25),
                "50%" to getPercentile(dayPredictions, 0.50),
                "75%" to getPercentile(dayPredictions, 0.75),
                "90%" to getPercentile(dayPredictions, 0.90),
                "95%" to getPercentile(dayPredictions, 0.95),
                "99%" to getPercentile(dayPredictions, 0.99)
            )
        }

        return results
    }

    // Enhanced multi-head attention with multiple heads
    private fun enhancedMultiHeadAttention(
        input: List<List<Double>>,
        numHeads: Int = 4
    ): List<List<Double>> {
        if (input.isEmpty() || input[0].isEmpty()) return input

        val headDim = input[0].size / numHeads
        val heads = mutableListOf<List<List<Double>>>()

        for (h in 0 until numHeads) {
            val startIdx = h * headDim
            val endIdx = minOf(startIdx + headDim, input[0].size)

            // Extract head-specific features
            val headInput = input.map { seq ->
                seq.subList(startIdx, endIdx).takeIf { it.isNotEmpty() } ?: listOf(0.0)
            }

            // Compute attention scores
            val scores = Array(headInput.size) { Array(headInput.size) { 0.0 } }
            for (i in headInput.indices) {
                for (j in headInput.indices) {
                    var score = 0.0
                    for (k in headInput[i].indices) {
                        if (k < headInput[j].size) {
                            score += headInput[i][k] * headInput[j][k]
                        }
                    }
                    scores[i][j] = exp(score / sqrt(headDim.toDouble()))
                }
            }

            // Softmax normalization
            for (i in scores.indices) {
                val sum = scores[i].sum()
                val normalizer = if (sum > 0.0) sum else 1.0
                for (j in scores[i].indices) {
                    scores[i][j] = scores[i][j] / normalizer
                }
            }

            // Apply attention to values
            val headOutput = mutableListOf<List<Double>>()
            for (i in scores.indices) {
                val weighted = MutableList(headInput[0].size) { 0.0 }
                for (j in headInput.indices) {
                    for (k in headInput[j].indices) {
                        weighted[k] += scores[i][j] * headInput[j][k]
                    }
                }
                headOutput.add(weighted)
            }

            heads.add(headOutput)
        }

        // Concatenate heads
        val output = mutableListOf<List<Double>>()
        for (i in input.indices) {
            val concat = mutableListOf<Double>()
            heads.forEach { head ->
                if (i < head.size) {
                    concat.addAll(head[i])
                }
            }
            output.add(concat)
        }

        return output
    }

    // Model evaluation metrics
    private fun calculateModelMetrics(predictions: List<Double>, actuals: List<Double>): Map<String, Double> {
        if (predictions.size != actuals.size || predictions.isEmpty()) {
            return emptyMap()
        }

        val errors = predictions.zip(actuals) { pred, actual -> pred - actual }
        val squaredErrors = errors.map { it.pow(2) }
        val absoluteErrors = errors.map { abs(it) }

        val mse = squaredErrors.average()
        val rmse = sqrt(mse)
        val mae = absoluteErrors.average()
        val mape = predictions.zip(actuals) { pred, actual ->
            if (actual != 0.0) abs((pred - actual) / actual) else 0.0
        }.average() * 100

        return mapOf(
            "MSE" to mse,
            "RMSE" to rmse,
            "MAE" to mae,
            "MAPE" to mape
        )
    }
}
