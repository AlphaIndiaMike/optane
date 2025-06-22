package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import android.util.Log

/**
 * TransformerForecaster - Mathematically Sound Implementation
 * Multi-head attention mechanism with deterministic feature engineering
 */
class TransformerForecasterImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

    // Algorithm-specific configuration
    private data class TransformerConfig(
        val sequenceLength: Int = 8,
        val hiddenDim: Int = 64,
        val numHeads: Int = 4,
        val numLayers: Int = 2,
        val maxHistoryDays: Int = 500 // Limit for performance
    )

    // FIXED: Deterministic features based only on price data
    private data class Features(
        val price: Double,
        val returns: Double,
        val logReturns: Double,
        val volatility: Double,
        val momentum: Double,
        val rsi: Double,
        val priceMA: Double,
        val volumeProxy: Double,      // FIXED: Deterministic volume proxy
        val bollingerPosition: Double, // Additional feature
        val trendStrength: Double     // Additional feature
    )

    private data class Scalers(
        val mean: Double,
        val std: Double
    )

    private val config = TransformerConfig()
    private lateinit var features: List<Features>
    private lateinit var scalers: Map<String, Scalers>

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < config.sequenceLength + 5) {
            return "Insufficient data"
        }

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        try {
            // FIXED: Limit data for performance
            val limitedTimeSeries = if (timeSeries.size > config.maxHistoryDays) {
                if (enableDebugLogging == true){
                Log.d("TransformerForecaster", "Limiting data from ${timeSeries.size} to ${config.maxHistoryDays} days")}
                timeSeries.takeLast(config.maxHistoryDays)
            } else {
                timeSeries
            }

            // 1. Extract deterministic features
            features = extractDeterministicFeatures(limitedTimeSeries)

            if (features.size < config.sequenceLength) {
                return "Unable to extract sufficient features"
            }

            // 2. Fit feature scalers
            scalers = fitScalers(features)

            // 3. Scale features
            val scaledFeatures = scaleFeatures(features)

            // 4. Apply transformer attention mechanism
            val transformerOutput = applyTransformerArchitecture(scaledFeatures)

            // 5. Generate deterministic forecasts with uncertainty quantification
            val result = generateForecastProbabilities(
                currentPrice = limitedTimeSeries.last().price,
                upperBand = upperBand,
                lowerBand = lowerBand,
                daysAhead = daysAhead,
                transformerOutput = transformerOutput,
                originalFeatures = features
            )

            if (enableDebugLogging) {
                logDebugInfo(limitedTimeSeries, result, daysAhead)
            }

            return """
                Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", result.first)}%
                Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", result.second)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("TransformerForecaster", "Error in calculation: ${e.message}")
            return "Calculation error: ${e.message}"
        }
    }

    // FIXED: Extract deterministic features from price data only
    private fun extractDeterministicFeatures(timeSeries: List<TimeSeriesEntity>): List<Features> {
        val features = mutableListOf<Features>()
        val prices = timeSeries.map { it.price }

        for (i in 5 until prices.size) { // Start from index 5 to have enough history
            try {
                val curr = prices[i]
                val prev = prices[i-1]

                // Basic price-based features
                val returns = (curr - prev) / prev
                val logReturns = ln(curr / prev)

                // Technical indicators
                val volatility = calculateRollingVolatility(prices, i, 5)
                val momentum = calculateMomentum(prices, i, 3)
                val rsi = calculateRSI(prices, i, 5)
                val priceMA = calculateMA(prices, i, 5)
                val volumeProxy = calculateVolumeProxy(prices, i, 5) // FIXED: Deterministic
                val bollingerPosition = calculateBollingerPosition(prices, i, 5)
                val trendStrength = calculateTrendStrength(prices, i, 5)

                features.add(Features(
                    price = curr,
                    returns = returns,
                    logReturns = logReturns,
                    volatility = volatility,
                    momentum = momentum,
                    rsi = rsi,
                    priceMA = priceMA,
                    volumeProxy = volumeProxy,
                    bollingerPosition = bollingerPosition,
                    trendStrength = trendStrength
                ))
            } catch (e: Exception) {
                Log.w("TransformerForecaster", "Skipping feature extraction at index $i: ${e.message}")
                continue
            }
        }

        return features
    }

    // FIXED: Deterministic volume proxy based on price action
    private fun calculateVolumeProxy(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 1000.0

        // Volume proxy based on price volatility and momentum
        val volatility = calculateRollingVolatility(prices, index, window)
        val momentum = abs(calculateMomentum(prices, index, window))

        // Higher volatility and momentum typically correlate with higher volume
        val baseVolume = 1000.0
        val volatilityFactor = volatility * 50000.0
        val momentumFactor = momentum * 10000.0

        return baseVolume + volatilityFactor + momentumFactor
    }

    private fun calculateRollingVolatility(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.02 // Default 2% daily volatility

        val returns = mutableListOf<Double>()
        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) {
                val ret = ln(prices[i] / prices[i-1])
                if (ret.isFinite()) {
                    returns.add(ret)
                }
            }
        }

        if (returns.isEmpty()) return 0.02

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance) // Daily volatility
    }

    private fun calculateMomentum(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        return (prices[index] - prices[index - window]) / prices[index - window]
    }

    private fun calculateRSI(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 50.0

        var gains = 0.0
        var losses = 0.0

        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) {
                val change = prices[i] - prices[i-1]
                if (change > 0) gains += change
                else losses += abs(change)
            }
        }

        val rs = gains / (losses.takeIf { it > 0.0 } ?: 1.0)
        return 100 - (100 / (1 + rs))
    }

    private fun calculateMA(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window - 1) return prices[index]

        var sum = 0.0
        for (i in (index - window + 1)..index) {
            sum += prices[i]
        }
        return sum / window
    }

    private fun calculateBollingerPosition(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.5

        val ma = calculateMA(prices, index, window)
        val volatility = calculateRollingVolatility(prices, index, window)
        val price = prices[index]

        val upperBand = ma + (2 * volatility * ma)
        val lowerBand = ma - (2 * volatility * ma)
        val range = upperBand - lowerBand

        return if (range > 0) (price - lowerBand) / range else 0.5
    }

    private fun calculateTrendStrength(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0

        val returns = mutableListOf<Double>()
        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) {
                returns.add(prices[i] - prices[i-1])
            }
        }

        if (returns.isEmpty()) return 0.0

        val posCount = returns.count { it > 0 }
        return (posCount.toDouble() / returns.size - 0.5) * 2 // Range: [-1, 1]
    }

    // Fit scalers for feature normalization
    private fun fitScalers(features: List<Features>): Map<String, Scalers> {
        if (features.isEmpty()) return emptyMap()

        val featureArrays = mapOf(
            "returns" to features.map { it.returns },
            "logReturns" to features.map { it.logReturns },
            "volatility" to features.map { it.volatility },
            "momentum" to features.map { it.momentum },
            "rsi" to features.map { it.rsi },
            "bollingerPosition" to features.map { it.bollingerPosition },
            "trendStrength" to features.map { it.trendStrength },
            "volumeProxy" to features.map { it.volumeProxy }
        )

        return featureArrays.mapValues { (_, values) ->
            val mean = values.average()
            val variance = values.map { (it - mean).pow(2) }.average()
            val std = sqrt(variance).takeIf { it > 0.0 } ?: 1.0
            Scalers(mean = mean, std = std)
        }
    }

    // Scale features using fitted scalers
    private fun scaleFeatures(features: List<Features>): List<List<Double>> {
        return features.map { feature ->
            listOf(
                scaleValue(feature.returns, "returns"),
                scaleValue(feature.logReturns, "logReturns"),
                scaleValue(feature.volatility, "volatility"),
                scaleValue(feature.momentum, "momentum"),
                scaleValue(feature.rsi, "rsi"),
                scaleValue(feature.bollingerPosition, "bollingerPosition"),
                scaleValue(feature.trendStrength, "trendStrength"),
                scaleValue(feature.volumeProxy, "volumeProxy")
            )
        }
    }

    private fun scaleValue(value: Double, featureName: String): Double {
        val scaler = scalers[featureName] ?: return value
        val scaled = (value - scaler.mean) / scaler.std
        return if (scaled.isFinite()) scaled else 0.0
    }

    // FIXED: Mathematically sound transformer architecture
    private fun applyTransformerArchitecture(scaledFeatures: List<List<Double>>): List<List<Double>> {
        if (scaledFeatures.isEmpty()) return emptyList()

        var currentRepresentation = scaledFeatures

        // Apply multiple transformer layers
        repeat(config.numLayers) { layer ->
            currentRepresentation = transformerLayer(currentRepresentation, layer)
        }

        return currentRepresentation
    }

    private fun transformerLayer(input: List<List<Double>>, layerIndex: Int): List<List<Double>> {
        // 1. Multi-head self-attention
        val attended = multiHeadSelfAttention(input, config.numHeads)

        // 2. Add & Norm (residual connection + layer normalization)
        val residual1 = addAndNorm(input, attended)

        // 3. Feed-forward network
        val feedForward = feedForwardNetwork(residual1, layerIndex)

        // 4. Add & Norm
        val residual2 = addAndNorm(residual1, feedForward)

        return residual2
    }

    private fun multiHeadSelfAttention(input: List<List<Double>>, numHeads: Int): List<List<Double>> {
        if (input.isEmpty() || input[0].isEmpty()) return input

        val featureDim = input[0].size
        val headDim = maxOf(1, featureDim / numHeads)
        val actualNumHeads = minOf(numHeads, featureDim)

        val headOutputs = mutableListOf<List<List<Double>>>()

        // Process each attention head
        for (h in 0 until actualNumHeads) {
            val startIdx = h * headDim
            val endIdx = minOf(startIdx + headDim, featureDim)

            // Extract features for this head
            val headInput = input.map { seq -> seq.subList(startIdx, endIdx) }

            // Compute attention scores
            val scores = Array(headInput.size) { Array(headInput.size) { 0.0 } }

            for (i in headInput.indices) {
                for (j in headInput.indices) {
                    var score = 0.0
                    for (k in headInput[i].indices) {
                        score += headInput[i][k] * headInput[j][k]
                    }
                    // Apply scaling and position bias
                    val scaledScore = score / sqrt(headDim.toDouble())
                    val positionBias = exp(-abs(i - j) * 0.1) // Exponential position decay
                    scores[i][j] = scaledScore * positionBias
                }
            }

            // Softmax normalization
            for (i in scores.indices) {
                val maxScore = scores[i].maxOrNull() ?: 0.0
                var sum = 0.0
                for (j in scores[i].indices) {
                    scores[i][j] = exp(scores[i][j] - maxScore) // Numerical stability
                    sum += scores[i][j]
                }
                if (sum > 0.0) {
                    for (j in scores[i].indices) {
                        scores[i][j] /= sum
                    }
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

            headOutputs.add(headOutput)
        }

        // Concatenate head outputs
        val concatenated = mutableListOf<List<Double>>()
        for (i in input.indices) {
            val concat = mutableListOf<Double>()
            headOutputs.forEach { headOutput ->
                if (i < headOutput.size) {
                    concat.addAll(headOutput[i])
                }
            }
            // Ensure output has same dimension as input
            while (concat.size < featureDim) concat.add(0.0)
            if (concat.size > featureDim) {
                concatenated.add(concat.take(featureDim))
            } else {
                concatenated.add(concat)
            }
        }

        return concatenated
    }

    private fun addAndNorm(input1: List<List<Double>>, input2: List<List<Double>>): List<List<Double>> {
        if (input1.size != input2.size) return input1

        return input1.zip(input2) { seq1, seq2 ->
            if (seq1.size != seq2.size) return@zip seq1

            // Add residual connection
            val added = seq1.zip(seq2) { a, b -> a + b }

            // Layer normalization
            val mean = added.average()
            val variance = added.map { (it - mean).pow(2) }.average()
            val std = sqrt(variance + 1e-8) // Small epsilon for numerical stability

            added.map { (it - mean) / std }
        }
    }

    private fun feedForwardNetwork(input: List<List<Double>>, layerIndex: Int): List<List<Double>> {
        return input.map { sequence ->
            // Simple feed-forward: expand -> activate -> contract
            val expanded = sequence.map { x ->
                // Linear transformation with layer-specific weights (deterministic)
                val weight = sin(layerIndex * 0.1 + 1.0) // Deterministic weights
                val bias = cos(layerIndex * 0.1)
                x * weight + bias
            }

            // ReLU activation
            val activated = expanded.map { maxOf(0.0, it) }

            // Contract back to original dimension
            activated.mapIndexed { index, value ->
                val weight2 = cos(layerIndex * 0.1 + index * 0.01 + 1.0)
                value * weight2
            }
        }
    }

    // FIXED: Generate deterministic forecast probabilities
    private fun generateForecastProbabilities(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        transformerOutput: List<List<Double>>,
        originalFeatures: List<Features>
    ): Pair<Double, Double> {

        // Extract signal from transformer output
        val recentOutput = transformerOutput.takeLast(config.sequenceLength)
        val transformerSignal = extractTransformerSignal(recentOutput)

        // Calculate historical statistics for scaling
        val historicalReturns = originalFeatures.map { it.logReturns }
        val meanReturn = historicalReturns.average()
        val volatility = sqrt(historicalReturns.map { (it - meanReturn).pow(2) }.average())

        // Generate deterministic price distribution using transformer signal
        val priceForecast = generateDeterministicPriceDistribution(
            currentPrice = currentPrice,
            transformerSignal = transformerSignal,
            meanReturn = meanReturn,
            volatility = volatility,
            daysAhead = daysAhead
        )

        // Calculate barrier probabilities
        val upperProbability = calculateBarrierProbability(
            targetPrice = upperBand,
            currentPrice = currentPrice,
            priceForecast = priceForecast,
            volatility = volatility,
            daysAhead = daysAhead,
            isUpper = true
        )

        val lowerProbability = calculateBarrierProbability(
            targetPrice = lowerBand,
            currentPrice = currentPrice,
            priceForecast = priceForecast,
            volatility = volatility,
            daysAhead = daysAhead,
            isUpper = false
        )

        return Pair(upperProbability * 100, lowerProbability * 100)
    }

    private fun extractTransformerSignal(transformerOutput: List<List<Double>>): Double {
        if (transformerOutput.isEmpty()) return 0.0

        // Extract directional signal from transformer output
        val flatOutput = transformerOutput.flatten()
        if (flatOutput.isEmpty()) return 0.0

        // Calculate weighted signal (emphasize recent outputs)
        var weightedSum = 0.0
        var totalWeight = 0.0

        flatOutput.forEachIndexed { index, value ->
            val weight = exp(index * 0.1) // Exponential weighting favoring later elements
            weightedSum += value * weight
            totalWeight += weight
        }

        val signal = if (totalWeight > 0) weightedSum / totalWeight else 0.0

        // Clamp signal to reasonable range
        return maxOf(-2.0, minOf(2.0, signal))
    }

    private fun generateDeterministicPriceDistribution(
        currentPrice: Double,
        transformerSignal: Double,
        meanReturn: Double,
        volatility: Double,
        daysAhead: Int
    ): Map<Double, Double> {

        // Adjust mean return based on transformer signal
        val adjustedMeanReturn = meanReturn + (transformerSignal * volatility * 0.1)

        // Generate quantile-based price distribution
        val quantiles = listOf(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

        return quantiles.associateWith { q ->
            // Use inverse normal approximation
            val zScore = getInverseNormal(q)
            val totalReturn = adjustedMeanReturn * daysAhead + zScore * volatility * sqrt(daysAhead.toDouble())
            currentPrice * exp(totalReturn)
        }
    }

    private fun getInverseNormal(p: Double): Double {
        if (p <= 0.0) return -3.0
        if (p >= 1.0) return 3.0

        // Beasley-Springer-Moro approximation
        val a0 = 2.515517
        val a1 = 0.802853
        val a2 = 0.010328
        val b1 = 1.432788
        val b2 = 0.189269
        val b3 = 0.001308

        val t = if (p <= 0.5) {
            sqrt(-2.0 * ln(p))
        } else {
            sqrt(-2.0 * ln(1.0 - p))
        }

        val numerator = a0 + a1 * t + a2 * t * t
        val denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t
        val result = t - numerator / denominator

        return if (p <= 0.5) -result else result
    }

    private fun calculateBarrierProbability(
        targetPrice: Double,
        currentPrice: Double,
        priceForecast: Map<Double, Double>,
        volatility: Double,
        daysAhead: Int,
        isUpper: Boolean
    ): Double {

        // Get end-point probability through interpolation
        val sortedQuantiles = priceForecast.keys.sorted()
        val sortedPrices = sortedQuantiles.map { priceForecast[it]!! }

        val endPointProb = interpolateProbability(targetPrice, sortedPrices, sortedQuantiles, isUpper)

        if (endPointProb <= 0.0 || endPointProb >= 1.0) return endPointProb

        // Apply barrier adjustment using reflection principle
        val logDistance = abs(ln(targetPrice / currentPrice))
        val timeScaledVol = volatility * sqrt(daysAhead.toDouble())

        val d1 = logDistance / timeScaledVol

        val barrierProbability = if (d1 > 3.0) {
            // Far barriers: use end-point probability
            endPointProb
        } else {
            // Near barriers: apply reflection principle
            val touchProbability = 2.0 * endPointProb
            minOf(1.0, touchProbability)
        }

        return barrierProbability
    }

    private fun interpolateProbability(
        targetPrice: Double,
        sortedPrices: List<Double>,
        sortedQuantiles: List<Double>,
        isUpper: Boolean
    ): Double {

        // Find bracketing quantiles
        for (i in 0 until sortedPrices.size - 1) {
            if (targetPrice >= sortedPrices[i] && targetPrice <= sortedPrices[i + 1]) {
                val weight = (targetPrice - sortedPrices[i]) / (sortedPrices[i + 1] - sortedPrices[i])
                val interpolatedQuantile = sortedQuantiles[i] + weight * (sortedQuantiles[i + 1] - sortedQuantiles[i])

                return if (isUpper) {
                    1.0 - interpolatedQuantile
                } else {
                    interpolatedQuantile
                }
            }
        }

        // Handle edge cases
        return if (isUpper) {
            if (targetPrice > sortedPrices.last()) 0.0 else 1.0
        } else {
            if (targetPrice < sortedPrices.first()) 0.0 else 1.0
        }
    }

    // Debug logging
    private fun logDebugInfo(
        timeSeries: List<TimeSeriesEntity>,
        result: Pair<Double, Double>,
        daysAhead: Int
    ) {
        Log.d("TransformerForecaster", "=== TRANSFORMER DEBUG INFO ===")
        Log.d("TransformerForecaster", "Data points: ${timeSeries.size}")
        Log.d("TransformerForecaster", "Features extracted: ${features.size}")
        Log.d("TransformerForecaster", "Days ahead: $daysAhead")
        Log.d("TransformerForecaster", "Current price: ${timeSeries.last().price}")
        Log.d("TransformerForecaster", "Upper probability: ${String.format("%.1f", result.first)}%")
        Log.d("TransformerForecaster", "Lower probability: ${String.format("%.1f", result.second)}%")
        Log.d("TransformerForecaster", "Config: sequence=${config.sequenceLength}, heads=${config.numHeads}, layers=${config.numLayers}")
        Log.d("TransformerForecaster", "=== END DEBUG INFO ===")
    }
}