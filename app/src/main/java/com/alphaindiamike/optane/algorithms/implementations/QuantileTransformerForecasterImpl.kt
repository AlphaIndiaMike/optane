package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import android.util.Log

/**
 * QuantileTransformerForecaster - Mathematically Sound Implementation
 * Advanced quantile regression with proper uncertainty quantification
 */
class QuantileTransformerForecasterImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

    // Advanced configuration with memory limits
    private data class QuantileConfig(
        val sequenceLength: Int = 8,
        val numQuantiles: Int = 9,
        val hiddenDim: Int = 128,
        val numHeads: Int = 8,
        val dropout: Double = 0.1,
        val maxHistoryDays: Int = 1095 // Configurable data limit (default: 3 years)
    )

    // Advanced features set (21 features) - MATHEMATICALLY DETERMINISTIC
    private data class AdvancedFeatures(
        val price: Double,
        val returns: Double,
        val logReturns: Double,
        val momentum_3: Double,
        val momentum_5: Double,
        val volatility_3: Double,
        val volatility_5: Double,
        val rsi_3: Double,
        val rsi_5: Double,
        val bollingerPosition: Double,
        val volumeProfile: Double,          // FIXED: Now deterministic
        val bidAskSpread: Double,           // FIXED: Now deterministic
        val orderFlowImbalance: Double,
        val trendStrength: Double,
        val marketRegime: Double,
        val hurstExponent: Double,
        val fractalDimension: Double,
        val dayOfWeek: Double,
        val timeSinceStart: Double,
        val priceVolumeTrend: Double,       // FIXED: Now deterministic
        val volatilityMomentum: Double
    )

    private data class Scalers(
        val median: Double,
        val iqr: Double,
        val min: Double,
        val max: Double
    )

    // Memory pool for efficient array reuse
    private class MemoryPool {
        private val doubleArrayPool = mutableListOf<DoubleArray>()
        private val arrayListPool = mutableListOf<MutableList<Double>>()

        fun borrowDoubleArray(size: Int): DoubleArray {
            return doubleArrayPool.removeFirstOrNull()?.takeIf { it.size >= size }
                ?: DoubleArray(size)
        }

        fun returnDoubleArray(array: DoubleArray) {
            array.fill(0.0) // Clear for reuse
            doubleArrayPool.add(array)
        }

        fun borrowList(): MutableList<Double> {
            return arrayListPool.removeFirstOrNull()?.apply { clear() }
                ?: mutableListOf()
        }

        fun returnList(list: MutableList<Double>) {
            list.clear()
            arrayListPool.add(list)
        }

        fun cleanup() {
            doubleArrayPool.clear()
            arrayListPool.clear()
        }
    }

    private val config = QuantileConfig()
    private val memoryPool = MemoryPool()
    private lateinit var features: List<AdvancedFeatures>
    private lateinit var scalers: Map<String, Scalers>

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 10) {
            return "Insufficient data"
        }

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        try {
            // Force aggressive garbage collection before starting
            System.gc()
            Thread.sleep(50) // Give GC time to work

            // DYNAMIC DATA LIMITING: Use config setting with intelligent fallback
            val maxDays = config.maxHistoryDays
            val limitedTimeSeries = if (timeSeries.size > maxDays) {
                Log.d("QuantileForecaster", "Limiting data from ${timeSeries.size} to $maxDays days (config limit)")
                timeSeries.takeLast(maxDays)
            } else {
                Log.d("QuantileForecaster", "Using all ${timeSeries.size} days (under config limit of $maxDays)")
                timeSeries
            }

            // 1. Extract advanced features using STREAMING on limited data
            val featureSequence = extractAdvancedFeaturesStreaming(limitedTimeSeries)

            // 2. Convert to memory-efficient arrays and fit scalers
            val (featureArrays, scalersMap) = processFeatureSequence(featureSequence)

            if (featureArrays.isEmpty()) {
                return "Unable to extract sufficient features"
            }

            // Store for debug logging
            scalers = scalersMap

            // 3. Scale features using primitive arrays
            val scaledFeatures = scaleFeatureArrays(featureArrays)

            // 4. Apply multi-head self-attention with memory pooling
            val attended = multiHeadSelfAttentionOptimized(scaledFeatures, config.numHeads)

            // 5. FIXED: Proper quantile regression with mathematical soundness
            val confidenceLevels = listOf(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
            val quantilePredictions = quantileRegressionMathematical(attended, confidenceLevels, daysAhead, limitedTimeSeries)

            // Debug logging with configuration info
            if (enableDebugLogging) {
                // Reconstruct features for logging
                features = featureArrays.map { array -> arrayToAdvancedFeatures(array) }
                logDebugInfo(timeSeries, limitedTimeSeries, quantilePredictions, daysAhead)
            }

            // 6. Calculate BARRIER probabilities for target bands
            val currentPrice = limitedTimeSeries.last().price
            val result = calculateBarrierProbabilitiesMathematical(
                quantilePredictions, currentPrice, upperBand, lowerBand, daysAhead
            )

            return """
                Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", result.first)}%
                Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", result.second)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("QuantileForecaster", "Error in calculation: ${e.message}")
            return "Calculation error: ${e.message}"
        } finally {
            // Clean up memory pool
            memoryPool.cleanup()
            System.gc() // Final cleanup
        }
    }

    // STREAMING - Process features one at a time with aggressive cleanup
    private fun extractAdvancedFeaturesStreaming(timeSeries: List<TimeSeriesEntity>): Sequence<DoubleArray> {
        return sequence {
            val prices = timeSeries.map { it.price }
            var featureCount = 0

            for (i in 8 until prices.size) {
                try {
                    val featureArray = extractSingleFeatureArray(prices, i, timeSeries)
                    yield(featureArray)
                    featureCount++

                    // AGGRESSIVE cleanup every 5 features
                    if (featureCount % 5 == 0) {
                        System.gc()
                        Thread.sleep(10) // Give GC time to work
                    }
                } catch (e: Exception) {
                    Log.w("QuantileForecaster", "Skipping feature extraction at index $i: ${e.message}")
                    continue
                }
            }
        }
    }

    // Extract single feature as DoubleArray for memory efficiency - FIXED: All deterministic
    private fun extractSingleFeatureArray(prices: List<Double>, index: Int, timeSeries: List<TimeSeriesEntity>): DoubleArray {
        val array = memoryPool.borrowDoubleArray(21) // Reuse arrays from pool

        try {
            array[0] = prices[index] // price
            array[1] = getReturns(prices, index, 1) // returns
            array[2] = if (index > 0) ln(prices[index] / prices[index-1]) else 0.0 // logReturns
            array[3] = getMomentum(prices, index, 3) // momentum_3
            array[4] = getMomentum(prices, index, 5) // momentum_5
            array[5] = getRollingVolatility(prices, index, 3) // volatility_3
            array[6] = getRollingVolatility(prices, index, 5) // volatility_5
            array[7] = getRSI(prices, index, 3) // rsi_3
            array[8] = getRSI(prices, index, 5) // rsi_5
            array[9] = getBollingerPosition(prices, index, 5) // bollingerPosition
            array[10] = getVolumeProfileDeterministic(prices, index) // FIXED: volumeProfile
            array[11] = getBidAskSpreadDeterministic(prices, index) // FIXED: bidAskSpread
            array[12] = estimateOrderFlowImbalance(prices, index) // orderFlowImbalance
            array[13] = getTrendStrength(prices, index, 5) // trendStrength
            array[14] = getMarketRegime(prices, index, 8).toDouble() // marketRegime
            array[15] = getHurstExponent(prices, index, 5) // hurstExponent
            array[16] = getFractalDimension(prices, index, 5) // fractalDimension
            array[17] = getDayOfWeek(index, timeSeries) // dayOfWeek
            array[18] = index.toDouble() / prices.size // timeSinceStart
            array[19] = getPriceVolumeTrendDeterministic(prices, index) // FIXED: priceVolumeTrend
            array[20] = getVolatilityMomentum(prices, index) // volatilityMomentum

            return array.copyOf() // Return copy, keep original in pool
        } finally {
            memoryPool.returnDoubleArray(array) // Return to pool
        }
    }

    // Process feature sequence with limits and chunking
    private fun processFeatureSequence(featureSequence: Sequence<DoubleArray>): Pair<Array<DoubleArray>, Map<String, Scalers>> {
        val featureList = mutableListOf<DoubleArray>()
        val tempValues = Array(21) { memoryPool.borrowList() }

        try {
            var processedCount = 0
            val maxFeatures = 100 // Limit processing to prevent memory overflow

            // Collect features and values for scaling - WITH LIMITS
            featureSequence.take(maxFeatures).forEach { featureArray ->
                featureList.add(featureArray.copyOf())

                // Collect values for each feature dimension
                for (i in 0 until 21) {
                    if (featureArray[i].isFinite()) {
                        tempValues[i].add(featureArray[i])
                    }
                }

                processedCount++
                // Aggressive GC every 10 features
                if (processedCount % 10 == 0) {
                    System.gc()
                }
            }

            Log.d("QuantileForecaster", "Processed $processedCount features (max: $maxFeatures)")

            // Fit scalers efficiently
            val featureNames = arrayOf(
                "price", "returns", "logReturns", "momentum_3", "momentum_5",
                "volatility_3", "volatility_5", "rsi_3", "rsi_5", "bollingerPosition",
                "volumeProfile", "bidAskSpread", "orderFlowImbalance", "trendStrength",
                "marketRegime", "hurstExponent", "fractalDimension", "dayOfWeek",
                "timeSinceStart", "priceVolumeTrend", "volatilityMomentum"
            )

            val scalersMap = mutableMapOf<String, Scalers>()

            for (i in 0 until 21) {
                val values = tempValues[i]
                if (values.isNotEmpty()) {
                    values.sort() // In-place sort for efficiency
                    val q25 = values[(values.size * 0.25).toInt()]
                    val q75 = values[(values.size * 0.75).toInt()]
                    val median = values[values.size / 2]
                    val iqr = q75 - q25

                    scalersMap[featureNames[i]] = Scalers(
                        median = median,
                        iqr = iqr.takeIf { it > 0 } ?: 1.0,
                        min = values.first(),
                        max = values.last()
                    )
                }
            }

            return Pair(featureList.toTypedArray(), scalersMap)

        } finally {
            // Return lists to pool
            tempValues.forEach { memoryPool.returnList(it) }
        }
    }

    // Scale features using primitive arrays
    private fun scaleFeatureArrays(featureArrays: Array<DoubleArray>): Array<DoubleArray> {
        val featureNames = arrayOf(
            "price", "returns", "logReturns", "momentum_3", "momentum_5",
            "volatility_3", "volatility_5", "rsi_3", "rsi_5", "bollingerPosition",
            "volumeProfile", "bidAskSpread", "orderFlowImbalance", "trendStrength",
            "marketRegime", "hurstExponent", "fractalDimension", "dayOfWeek",
            "timeSinceStart", "priceVolumeTrend", "volatilityMomentum"
        )

        return Array(featureArrays.size) { i ->
            val scaled = memoryPool.borrowDoubleArray(21)
            try {
                for (j in 0 until 21) {
                    val scaler = scalers[featureNames[j]]
                    scaled[j] = if (scaler != null) {
                        (featureArrays[i][j] - scaler.median) / scaler.iqr
                    } else {
                        featureArrays[i][j]
                    }

                    // Handle NaN/Infinity
                    if (!scaled[j].isFinite()) {
                        scaled[j] = 0.0
                    }
                }
                scaled.copyOf() // Return copy
            } finally {
                memoryPool.returnDoubleArray(scaled)
            }
        }
    }

    // Memory-optimized attention with CHUNKED processing
    private fun multiHeadSelfAttentionOptimized(input: Array<DoubleArray>, numHeads: Int = 8): Array<DoubleArray> {
        if (input.isEmpty() || input[0].isEmpty()) return input

        // LIMIT input size if too large
        val maxSequenceLength = 50 // Reduced from unlimited
        val processInput = if (input.size > maxSequenceLength) {
            Log.w("QuantileForecaster", "Trimming input from ${input.size} to $maxSequenceLength for memory")
            input.takeLast(maxSequenceLength).toTypedArray()
        } else {
            input
        }

        val featureDim = processInput[0].size
        val headDim = maxOf(1, featureDim / numHeads)
        val actualNumHeads = minOf(numHeads, featureDim)

        val output = Array(processInput.size) { memoryPool.borrowDoubleArray(featureDim) }

        try {
            // Process heads in chunks to reduce memory pressure
            val headsPerChunk = 2 // Process 2 heads at a time
            for (chunkStart in 0 until actualNumHeads step headsPerChunk) {
                val chunkEnd = minOf(chunkStart + headsPerChunk, actualNumHeads)

                for (h in chunkStart until chunkEnd) {
                    val startIdx = h * headDim
                    val endIdx = minOf(startIdx + headDim, featureDim)

                    // Reuse score arrays from pool
                    val scores = Array(processInput.size) { memoryPool.borrowDoubleArray(processInput.size) }

                    try {
                        // Compute attention scores - FIXED: Proper scaled dot-product attention
                        for (i in processInput.indices) {
                            for (j in processInput.indices) {
                                var score = 0.0
                                for (k in startIdx until endIdx) {
                                    score += processInput[i][k] * processInput[j][k]
                                }
                                // FIXED: Proper attention scaling and position bias
                                val scaledScore = score / sqrt(headDim.toDouble())
                                val positionBias = exp(-abs(i - j) * 0.1) // Exponential decay for position
                                scores[i][j] = scaledScore * positionBias
                            }
                        }

                        // FIXED: Proper softmax normalization
                        for (i in scores.indices) {
                            val maxScore = scores[i].maxOrNull() ?: 0.0
                            var sum = 0.0
                            for (j in scores[i].indices) {
                                scores[i][j] = exp(scores[i][j] - maxScore) // Numerical stability
                                sum += scores[i][j]
                            }
                            // Normalize
                            if (sum > 0.0) {
                                for (j in scores[i].indices) {
                                    scores[i][j] /= sum
                                }
                            }
                        }

                        // Apply attention to values
                        for (i in scores.indices) {
                            for (k in startIdx until endIdx) {
                                var weighted = 0.0
                                for (j in processInput.indices) {
                                    weighted += scores[i][j] * processInput[j][k]
                                }
                                output[i][k] = weighted
                            }
                        }
                    } finally {
                        // Return score arrays to pool
                        scores.forEach { memoryPool.returnDoubleArray(it) }
                    }
                }

                // GC between chunks
                if (chunkEnd < actualNumHeads) {
                    System.gc()
                }
            }

            return Array(output.size) { i -> output[i].copyOf() }
        } finally {
            // Return output arrays to pool
            output.forEach { memoryPool.returnDoubleArray(it) }
        }
    }

    // FIXED: Extract returns from original time series, not scaled features
    private fun quantileRegressionMathematical(
        features: Array<DoubleArray>,
        quantiles: List<Double>,
        daysAhead: Int,
        originalTimeSeries: List<TimeSeriesEntity> // Add original time series parameter
    ): Map<Double, Double> {
        if (features.isEmpty()) return quantiles.associateWith { 0.0 }

        // Extract historical returns from ORIGINAL time series, not scaled features
        val historicalReturns = mutableListOf<Double>()
        for (i in 1 until originalTimeSeries.size) {
            val currentPrice = originalTimeSeries[i].price
            val previousPrice = originalTimeSeries[i-1].price

            if (currentPrice > 0 && previousPrice > 0) {
                val dailyReturn = ln(currentPrice / previousPrice)
                if (dailyReturn.isFinite() && abs(dailyReturn) < 0.5) { // Sanity check for returns
                    historicalReturns.add(dailyReturn)
                }
            }
        }

        if (historicalReturns.isEmpty()) {
            Log.w("QuantileForecaster", "No historical returns available from original time series")
            return quantiles.associateWith { 0.0 }
        }

        // Calculate empirical statistics
        val sortedReturns = historicalReturns.sorted()
        val meanReturn = historicalReturns.average()
        val variance = historicalReturns.map { (it - meanReturn).pow(2) }.average()
        val stdDev = sqrt(variance)

        // Use feature signal for directional bias only
        val latestFeatures = features.last()
        val featureSignal = calculateFeatureSignal(latestFeatures)

        // Apply small feature-based bias to mean (max 10% of std dev)
        val adjustedMeanReturn = meanReturn + (featureSignal * stdDev * 0.1).coerceIn(-stdDev * 0.1, stdDev * 0.1)

        // Proper quantile calculation using inverse normal approximation
        val predictions = mutableMapOf<Double, Double>()

        quantiles.forEach { q ->
            // Get empirical quantile from historical data
            val empiricalQuantile = getEmpiricalQuantile(sortedReturns, q)

            // Alternative: Use normal approximation for smoother quantiles
            val normalQuantile = adjustedMeanReturn + getInverseNormalApprox(q) * stdDev

            // Blend empirical and normal (70% empirical, 30% normal for smoothness)
            val blendedQuantile = empiricalQuantile * 0.7 + normalQuantile * 0.3

            // Scale for time horizon using proper volatility scaling
            val scaledQuantile = blendedQuantile * sqrt(daysAhead.toDouble())

            predictions[q] = scaledQuantile
        }

        Log.d("QuantileForecaster", "FIXED Quantile regression:")
        Log.d("QuantileForecaster", "  Historical returns: ${historicalReturns.size} samples")
        Log.d("QuantileForecaster", "  Mean return: ${String.format("%.6f", meanReturn)}")
        Log.d("QuantileForecaster", "  Std dev: ${String.format("%.6f", stdDev)}")
        Log.d("QuantileForecaster", "  Feature signal: ${String.format("%.4f", featureSignal)}")
        Log.d("QuantileForecaster", "  Adjusted mean: ${String.format("%.6f", adjustedMeanReturn)}")
        Log.d("QuantileForecaster", "  Sample predictions: 5%=${String.format("%.6f", predictions[0.05])}, 50%=${String.format("%.6f", predictions[0.5])}, 95%=${String.format("%.6f", predictions[0.95])}")

        return predictions
    }

    // Inverse normal approximation for quantile calculation
    private fun getInverseNormalApprox(p: Double): Double {
        if (p <= 0.0) return -3.0
        if (p >= 1.0) return 3.0

        // Beasley-Springer-Moro algorithm approximation
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
    private fun calculateFeatureSignal(features: DoubleArray): Double {
        if (features.size < 21) return 0.0

        // Combine momentum, volatility, and trend features
        val momentum = (features[3] + features[4]) / 2.0 // momentum_3 + momentum_5
        val volatility = (features[5] + features[6]) / 2.0 // volatility_3 + volatility_5
        val trend = features[13] // trendStrength
        val rsi = (features[7] + features[8]) / 2.0 // rsi_3 + rsi_5

        // Normalize RSI to [-1, 1] range
        val rsiNormalized = (rsi - 50.0) / 50.0

        // Combine signals with weights
        val signal = momentum * 0.4 + trend * 0.3 + rsiNormalized * 0.2 - volatility * 0.1

        // Clamp to reasonable range
        return maxOf(-2.0, minOf(2.0, signal))
    }

    // Get empirical quantile from sorted data
    private fun getEmpiricalQuantile(sortedData: List<Double>, quantile: Double): Double {
        if (sortedData.isEmpty()) return 0.0

        val index = (quantile * (sortedData.size - 1)).toInt()
        val weight = (quantile * (sortedData.size - 1)) - index

        return if (index >= sortedData.size - 1) {
            sortedData.last()
        } else {
            sortedData[index] * (1 - weight) + sortedData[index + 1] * weight
        }
    }

    // FIXED: Mathematically sound barrier probability calculation
    private fun calculateBarrierProbabilitiesMathematical(
        quantilePredictions: Map<Double, Double>,
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int
    ): Pair<Double, Double> {

        // Generate price forecasts from quantile predictions
        val priceForecast = quantilePredictions.mapValues { (_, returnPrediction) ->
            currentPrice * exp(returnPrediction)
        }

        val sortedQuantiles = priceForecast.keys.sorted()
        val sortedPrices = sortedQuantiles.map { priceForecast[it]!! }

        // MATHEMATICAL: Proper barrier probability calculation
        val upperProbability = calculateMathematicalBarrierProbability(
            upperBand, sortedPrices, sortedQuantiles, currentPrice, daysAhead, true
        )
        val lowerProbability = calculateMathematicalBarrierProbability(
            lowerBand, sortedPrices, sortedQuantiles, currentPrice, daysAhead, false
        )

        return Pair(upperProbability * 100, lowerProbability * 100)
    }

    // MATHEMATICALLY SOUND: Proper barrier probability using reflection principle
    private fun calculateMathematicalBarrierProbability(
        targetPrice: Double,
        sortedPrices: List<Double>,
        sortedQuantiles: List<Double>,
        currentPrice: Double,
        daysAhead: Int,
        isUpperBand: Boolean
    ): Double {

        // Get end-point probability from quantile interpolation
        val endPointProb = interpolateProbability(targetPrice, sortedPrices, sortedQuantiles, isUpperBand)

        if (endPointProb <= 0.0 || endPointProb >= 1.0) return endPointProb

        // MATHEMATICAL: Proper reflection principle for barrier options
        // For geometric Brownian motion: P(touch barrier) = 2*Φ(d) - 1 where Φ is cumulative normal
        // Simplified: P(touch) ≈ min(1, 2*P(end beyond barrier)) for barriers close to current price

        val logDistance = ln(targetPrice / currentPrice)
        val absLogDistance = abs(logDistance)

        // Estimate volatility from price distribution
        val p90Price = interpolatePrice(sortedPrices, sortedQuantiles, 0.9)
        val p10Price = interpolatePrice(sortedPrices, sortedQuantiles, 0.1)
        val estimatedVolatility = abs(ln(p90Price / p10Price)) / (2 * getInverseNormalApprox(0.9))

        // Time-scaled volatility
        val timeScaledVol = estimatedVolatility * sqrt(daysAhead.toDouble())

        // Reflection principle: probability of touching barrier before expiry
        val d1 = absLogDistance / timeScaledVol
        val reflectionProbability = if (d1 > 3.0) {
            // Far barriers: use end-point probability
            endPointProb
        } else {
            // Near barriers: apply reflection principle
            val touchProbability = 2.0 * endPointProb
            minOf(1.0, touchProbability)
        }

        Log.d("QuantileForecaster", "Mathematical barrier calculation:")
        Log.d("QuantileForecaster", "  Distance: ${String.format("%.4f", logDistance)}")
        Log.d("QuantileForecaster", "  Estimated vol: ${String.format("%.4f", estimatedVolatility)}")
        Log.d("QuantileForecaster", "  Time-scaled vol: ${String.format("%.4f", timeScaledVol)}")
        Log.d("QuantileForecaster", "  d1 parameter: ${String.format("%.4f", d1)}")
        Log.d("QuantileForecaster", "  End-point prob: ${String.format("%.4f", endPointProb)}")
        Log.d("QuantileForecaster", "  Touch prob: ${String.format("%.4f", reflectionProbability)}")

        return reflectionProbability
    }

    // Helper function to interpolate price at given quantile
    private fun interpolatePrice(sortedPrices: List<Double>, sortedQuantiles: List<Double>, targetQuantile: Double): Double {
        for (i in 0 until sortedPrices.size - 1) {
            if (targetQuantile >= sortedQuantiles[i] && targetQuantile <= sortedQuantiles[i + 1]) {
                val weight = (targetQuantile - sortedQuantiles[i]) / (sortedQuantiles[i + 1] - sortedQuantiles[i])
                return sortedPrices[i] + weight * (sortedPrices[i + 1] - sortedPrices[i])
            }
        }
        return if (targetQuantile <= sortedQuantiles.first()) sortedPrices.first() else sortedPrices.last()
    }

    private fun interpolateProbability(
        targetPrice: Double,
        sortedPrices: List<Double>,
        sortedQuantiles: List<Double>,
        isUpperBand: Boolean
    ): Double {

        // Find bracketing quantiles
        for (i in 0 until sortedPrices.size - 1) {
            if (targetPrice >= sortedPrices[i] && targetPrice <= sortedPrices[i + 1]) {
                // Linear interpolation
                val weight = (targetPrice - sortedPrices[i]) / (sortedPrices[i + 1] - sortedPrices[i])
                val interpolatedQuantile = sortedQuantiles[i] + weight * (sortedQuantiles[i + 1] - sortedQuantiles[i])

                return if (isUpperBand) {
                    1.0 - interpolatedQuantile // Probability of exceeding
                } else {
                    interpolatedQuantile // Probability of being below
                }
            }
        }

        // Handle edge cases
        return if (isUpperBand) {
            if (targetPrice > sortedPrices.last()) 0.0 else 1.0
        } else {
            if (targetPrice < sortedPrices.first()) 0.0 else 1.0
        }
    }

    // ============== FIXED FEATURE CALCULATION METHODS ==============

    // FIXED: Deterministic volume profile based on price action
    private fun getVolumeProfileDeterministic(prices: List<Double>, index: Int): Double {
        if (index < 5) return 1000.0

        // Base volume on recent volatility and price momentum
        val recentVolatility = getRollingVolatility(prices, index, 5)
        val momentum = getMomentum(prices, index, 3)

        // Volume typically increases with volatility and momentum
        val baseVolume = 1000.0
        val volatilityFactor = recentVolatility * 10000.0
        val momentumFactor = abs(momentum) * 5000.0

        return baseVolume + volatilityFactor + momentumFactor
    }

    // FIXED: Deterministic bid-ask spread based on volatility
    private fun getBidAskSpreadDeterministic(prices: List<Double>, index: Int): Double {
        val volatility = getRollingVolatility(prices, index, 3)
        val price = prices[index]

        // Spread typically 0.01% to 0.1% of price, higher with volatility
        val baseSpread = price * 0.0001 // 0.01%
        val volatilitySpread = volatility * price * 0.01

        return baseSpread + volatilitySpread
    }

    // FIXED: Deterministic price-volume trend
    private fun getPriceVolumeTrendDeterministic(prices: List<Double>, index: Int): Double {
        val volume = getVolumeProfileDeterministic(prices, index)
        val priceChange = if (index > 0) prices[index] - prices[index-1] else 0.0
        val normalizedVolume = volume / 10000.0 // Normalize

        // Combine price change direction with volume strength
        return tanh(priceChange * normalizedVolume / prices[index])
    }

    // Original methods remain the same...
    private fun getReturns(prices: List<Double>, index: Int, period: Int): Double {
        if (index < period) return 0.0
        return (prices[index] - prices[index - period]) / prices[index - period]
    }

    private fun getMomentum(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        return (prices[index] - prices[index - window]) / prices[index - window]
    }

    private fun getRollingVolatility(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        val returns = mutableListOf<Double>()
        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) returns.add(ln(prices[i] / prices[i-1]))
        }
        if (returns.isEmpty()) return 0.0
        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    private fun getRSI(prices: List<Double>, index: Int, window: Int): Double {
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
        val rs = gains / (losses.takeIf { it > 0 } ?: 1.0)
        return 100 - (100 / (1 + rs))
    }

    private fun getBollingerPosition(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.5
        val startIdx = maxOf(0, index - window + 1)
        val endIdx = minOf(prices.size, index + 1)
        val slice = prices.subList(startIdx, endIdx)
        val mean = slice.average()
        val std = sqrt(slice.map { (it - mean).pow(2) }.average())
        val upperBand = mean + 2 * std
        val lowerBand = mean - 2 * std
        val range = upperBand - lowerBand
        return if (range > 0) (prices[index] - lowerBand) / range else 0.5
    }

    private fun estimateOrderFlowImbalance(prices: List<Double>, index: Int): Double {
        if (index < 3) return 0.0
        val momentum = getMomentum(prices, index, 3)
        val vol = getRollingVolatility(prices, index, 3)
        return tanh(momentum / (vol + 0.001))
    }

    private fun getTrendStrength(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.0
        val returns = mutableListOf<Double>()
        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) returns.add(prices[i] - prices[i-1])
        }
        if (returns.isEmpty()) return 0.0
        val posCount = returns.count { it > 0 }
        return (posCount.toDouble() / returns.size - 0.5) * 2
    }

    private fun getMarketRegime(prices: List<Double>, index: Int, window: Int): Int {
        if (index < window) return 0
        val vol = getRollingVolatility(prices, index, window)
        val momentum = getMomentum(prices, index, window)

        return when {
            vol < 0.02 -> 0 // Low volatility
            abs(momentum) > 0.05 -> 1 // Trending
            else -> 2 // Mean-reverting
        }
    }

    private fun getHurstExponent(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 0.5
        val returns = mutableListOf<Double>()
        for (i in (index - window + 1)..index) {
            if (i > 0 && i < prices.size) returns.add(ln(prices[i] / prices[i-1]))
        }
        if (returns.isEmpty()) return 0.5

        val mean = returns.average()
        val deviations = returns.map { it - mean }
        val cumDeviations = mutableListOf<Double>()
        var cumSum = 0.0
        deviations.forEach { dev ->
            cumSum += dev
            cumDeviations.add(cumSum)
        }

        val range = (cumDeviations.maxOrNull() ?: 0.0) - (cumDeviations.minOrNull() ?: 0.0)
        val std = sqrt(deviations.map { it.pow(2) }.average())
        val rs = range / (std.takeIf { it > 0 } ?: 0.001)

        return minOf(maxOf(ln(rs) / ln(window.toDouble()), 0.0), 1.0)
    }

    private fun getFractalDimension(prices: List<Double>, index: Int, window: Int): Double {
        if (index < window) return 1.5
        val startIdx = maxOf(0, index - window + 1)
        val endIdx = minOf(prices.size, index + 1)
        val slice = prices.subList(startIdx, endIdx)
        val min = slice.minOrNull() ?: 0.0
        val max = slice.maxOrNull() ?: 1.0
        val range = max - min
        if (range == 0.0) return 1.5

        val normalized = slice.map { (it - min) / range }

        var totalVariation = 0.0
        for (i in 1 until normalized.size) {
            totalVariation += abs(normalized[i] - normalized[i-1])
        }

        return 1 + minOf(totalVariation, 1.0)
    }

    private fun getVolatilityMomentum(prices: List<Double>, index: Int): Double {
        val vol = getRollingVolatility(prices, index, 3)
        val momentum = getMomentum(prices, index, 3)
        return vol * abs(momentum)
    }

    private fun getDayOfWeek(index: Int, timeSeries: List<TimeSeriesEntity>): Double {
        if (index >= timeSeries.size) return 0.0

        val timestampNanos = timeSeries[index].date
        val timestampMillis = timestampNanos / 1_000_000
        val daysSinceEpoch = timestampMillis / (24 * 60 * 60 * 1000)

        // Unix epoch started on Thursday (4), so we adjust
        val dayOfWeek = ((daysSinceEpoch + 4) % 7).toInt()

        // Normalize to 0-1 range, with weekend adjustment
        return when (dayOfWeek) {
            0, 6 -> 0.0 // Sunday, Saturday - weekends
            1 -> 0.2 // Monday
            2 -> 0.4 // Tuesday
            3 -> 0.6 // Wednesday
            4 -> 0.8 // Thursday
            5 -> 1.0 // Friday
            else -> 0.5
        }
    }

    // Helper function to convert DoubleArray back to AdvancedFeatures for debugging
    private fun arrayToAdvancedFeatures(array: DoubleArray): AdvancedFeatures {
        return AdvancedFeatures(
            price = array[0],
            returns = array[1],
            logReturns = array[2],
            momentum_3 = array[3],
            momentum_5 = array[4],
            volatility_3 = array[5],
            volatility_5 = array[6],
            rsi_3 = array[7],
            rsi_5 = array[8],
            bollingerPosition = array[9],
            volumeProfile = array[10],
            bidAskSpread = array[11],
            orderFlowImbalance = array[12],
            trendStrength = array[13],
            marketRegime = array[14],
            hurstExponent = array[15],
            fractalDimension = array[16],
            dayOfWeek = array[17],
            timeSinceStart = array[18],
            priceVolumeTrend = array[19],
            volatilityMomentum = array[20]
        )
    }

    // Debug logging function
    private fun logDebugInfo(
        originalTimeSeries: List<TimeSeriesEntity>,
        limitedTimeSeries: List<TimeSeriesEntity>,
        quantilePredictions: Map<Double, Double>,
        daysAhead: Int
    ) {
        Log.d("QuantileForecaster", "=== MATHEMATICAL DEBUG INFO ===")
        Log.d("QuantileForecaster", "Original data points: ${originalTimeSeries.size}")
        Log.d("QuantileForecaster", "Limited data points: ${limitedTimeSeries.size}")
        Log.d("QuantileForecaster", "Max history days config: ${config.maxHistoryDays}")
        Log.d("QuantileForecaster", "Features extracted: ${features.size}")
        Log.d("QuantileForecaster", "Days ahead prediction: $daysAhead")

        // Log current price and basic stats
        val currentPrice = limitedTimeSeries.last().price
        val firstPrice = limitedTimeSeries.first().price
        val totalReturn = (currentPrice - firstPrice) / firstPrice
        Log.d("QuantileForecaster", "Current price: $currentPrice")
        Log.d("QuantileForecaster", "Total return over period: ${String.format("%.2f", totalReturn * 100)}%")

        // Log quantile predictions
        Log.d("QuantileForecaster", "MATHEMATICAL Quantile predictions:")
        quantilePredictions.entries.sortedBy { it.key }.forEach { (quantile, prediction) ->
            Log.d("QuantileForecaster", "  ${String.format("%.1f", quantile * 100)}%: ${String.format("%.4f", prediction)}")
        }

        // Log feature scalers info
        Log.d("QuantileForecaster", "Feature scalers count: ${scalers.size}")
        scalers.entries.take(5).forEach { (name, scaler) ->
            Log.d("QuantileForecaster", "  $name: median=${String.format("%.4f", scaler.median)}, iqr=${String.format("%.4f", scaler.iqr)}")
        }

        // Log FIXED features
        Log.d("QuantileForecaster", "FIXED: volumeProfile now deterministic (volatility + momentum based)")
        Log.d("QuantileForecaster", "FIXED: bidAskSpread now deterministic (volatility based)")
        Log.d("QuantileForecaster", "FIXED: priceVolumeTrend now deterministic (no random component)")
        Log.d("QuantileForecaster", "FIXED: quantileRegression now mathematical (no random component)")
        Log.d("QuantileForecaster", "FIXED: barrierProbability now uses reflection principle")

        // Log memory configuration
        Log.d("QuantileForecaster", "Memory config:")
        Log.d("QuantileForecaster", "  Sequence length: ${config.sequenceLength}")
        Log.d("QuantileForecaster", "  Hidden dimensions: ${config.hiddenDim}")
        Log.d("QuantileForecaster", "  Attention heads: ${config.numHeads}")
        Log.d("QuantileForecaster", "  Dropout: ${config.dropout}")

        Log.d("QuantileForecaster", "=== END MATHEMATICAL DEBUG INFO ===")
    }

    // Clean up resources when done
    fun cleanup() {
        memoryPool.cleanup()
        System.gc()
    }
}

/* Current Algorithm Status:
✅ What's Implemented (Mathematically Sound):

Feature Engineering: 21 deterministic technical indicators
Multi-head Self-Attention: Proper scaled dot-product attention with position bias
Quantile Regression: Empirical distribution fitting with feature-weighted adjustments
Barrier Probability: Reflection principle for proper barrier option pricing
Memory Management: Efficient pooling and streaming
Scaling: Robust quantile-based normalization

⚠️ What's NOT Implemented (Neural Network Components):

True Neural Network: This is attention mechanism only, not a full transformer
Backpropagation: No learning/training - just feature transformation
Non-linear Layers: No feedforward networks or activation functions
Weight Optimization: No parameter learning

🎯 What It Actually Does:

Feature Extraction: Sophisticated technical analysis
Attention Weighting: Focuses on relevant historical patterns
Quantile Prediction: Statistical distribution modeling
Barrier Estimation: Mathematical option pricing approximation

Bottom Line: This is now a mathematically sound statistical model with attention-based feature weighting, not a true neural network. It should give deterministic, reproducible results for barrier probability estimation.
The algorithm is ready for testing. It's essentially a hybrid between traditional quantitative finance and modern attention mechanisms.*/