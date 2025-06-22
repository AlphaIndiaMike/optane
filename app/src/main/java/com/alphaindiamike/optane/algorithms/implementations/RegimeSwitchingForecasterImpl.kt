package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import android.util.Log

/**
 * RegimeSwitchingForecaster - Mathematically Sound Implementation
 * Deterministic Markov regime-switching model with analytical probability calculation
 */
class RegimeSwitchingForecasterImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

    // FIXED: Comprehensive regime parameters
    private data class RegimeParameters(
        val mean: Double,
        val volatility: Double,
        val persistence: Double,      // How likely to stay in regime
        val unconditionalProb: Double // Long-run probability of regime
    )

    // Configuration for regime switching model
    private data class RegimeConfig(
        val numRegimes: Int = 3,
        val minObservationsPerRegime: Int = 5,
        val volatilityWindow: Int = 10,
        val trendWindow: Int = 15,
        val maxHistoryDays: Int = 300
    )

    private val config = RegimeConfig()

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // FIXED: Proper input validation
        if (timeSeries.size < 30) {
            return "Insufficient data"
        }

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        try {
            // FIXED: Limit data for performance and stability
            val limitedTimeSeries = if (timeSeries.size > config.maxHistoryDays) {
                if (enableDebugLogging == true){
                Log.d("RegimeSwitching", "Limiting data from ${timeSeries.size} to ${config.maxHistoryDays} days")}
                timeSeries.takeLast(config.maxHistoryDays)
            } else {
                timeSeries
            }

            // 1. Calculate returns
            val returns = calculateReturns(limitedTimeSeries)

            // 2. FIXED: Deterministic regime identification
            val regimes = identifyRegimesDeterministic(returns)

            // 3. Estimate transition matrix
            val transitionMatrix = estimateTransitionMatrix(regimes)

            // 4. Estimate regime parameters
            val regimeParameters = estimateRegimeParameters(returns, regimes)

            // 5. FIXED: Deterministic analytical forecasting
            val result = forecastAnalytical(
                currentPrice = limitedTimeSeries.last().price,
                currentRegime = regimes.last(),
                upperBand = upperBand,
                lowerBand = lowerBand,
                daysAhead = daysAhead,
                transitionMatrix = transitionMatrix,
                regimeParameters = regimeParameters
            )

            if (enableDebugLogging) {
                logDebugInfo(limitedTimeSeries, regimes, transitionMatrix, regimeParameters, result, daysAhead)
            }

            return """
                Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", result.first)}%
                Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", result.second)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("RegimeSwitching", "Error in calculation: ${e.message}")
            return "Calculation error: ${e.message}"
        }
    }

    private fun calculateReturns(timeSeries: List<TimeSeriesEntity>): List<Double> {
        val returns = mutableListOf<Double>()
        for (i in 1 until timeSeries.size) {
            val ret = ln(timeSeries[i].price / timeSeries[i-1].price)
            if (ret.isFinite()) {
                returns.add(ret)
            }
        }
        return returns
    }

    // FIXED: Deterministic regime identification using statistical measures
    private fun identifyRegimesDeterministic(returns: List<Double>): List<Int> {
        if (returns.size < config.volatilityWindow) {
            return List(returns.size) { 0 } // Default to regime 0
        }

        val regimes = mutableListOf<Int>()

        // Calculate adaptive thresholds based on data
        val overallVolatility = calculateVolatility(returns)
        val overallMean = returns.average()

        // Adaptive thresholds (deterministic)
        val lowVolThreshold = overallVolatility * 0.7
        val highVolThreshold = overallVolatility * 1.5
        val trendThreshold = abs(overallMean) * 2.0

        for (i in returns.indices) {
            val regime = if (i < config.volatilityWindow) {
                0 // Default regime for early observations
            } else {
                val windowStart = maxOf(0, i - config.volatilityWindow + 1)
                val recentReturns = returns.subList(windowStart, i + 1)
                val recentVol = calculateVolatility(recentReturns)
                val recentMean = recentReturns.average()

                // FIXED: Proper regime classification
                when {
                    recentVol <= lowVolThreshold -> 0  // Low volatility regime
                    recentVol >= highVolThreshold -> 1 // High volatility regime
                    abs(recentMean) >= trendThreshold -> 2 // Trending regime
                    else -> 0 // Default to low volatility
                }
            }
            regimes.add(regime)
        }

        return regimes
    }

    private fun calculateVolatility(returns: List<Double>): Double {
        if (returns.isEmpty()) return 0.02 // Default volatility

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    private fun estimateTransitionMatrix(regimes: List<Int>): Array<Array<Double>> {
        val numRegimes = config.numRegimes
        val transitions = Array(numRegimes) { Array(numRegimes) { 1.0 } } // Laplace smoothing

        // Count transitions
        for (i in 1 until regimes.size) {
            val from = regimes[i - 1]
            val to = regimes[i]
            if (from < numRegimes && to < numRegimes) {
                transitions[from][to]++
            }
        }

        // FIXED: Normalize to probabilities with smoothing
        for (i in 0 until numRegimes) {
            val sum = transitions[i].sum()
            if (sum > 0) {
                for (j in 0 until numRegimes) {
                    transitions[i][j] = transitions[i][j] / sum
                }
            } else {
                // Equal probabilities if no data
                for (j in 0 until numRegimes) {
                    transitions[i][j] = 1.0 / numRegimes
                }
            }
        }

        return transitions
    }

    private fun estimateRegimeParameters(returns: List<Double>, regimes: List<Int>): Map<Int, RegimeParameters> {
        val numRegimes = config.numRegimes
        val parameters = mutableMapOf<Int, RegimeParameters>()

        for (regime in 0 until numRegimes) {
            val regimeReturns = returns.filterIndexed { index, _ ->
                index < regimes.size && regimes[index] == regime
            }

            if (regimeReturns.size >= config.minObservationsPerRegime) {
                val mean = regimeReturns.average()
                val variance = regimeReturns.map { (it - mean).pow(2) }.average()
                val volatility = sqrt(variance)

                // Calculate regime persistence
                val persistence = calculateRegimePersistence(regimes, regime)
                val unconditionalProb = regimeReturns.size.toDouble() / returns.size

                parameters[regime] = RegimeParameters(
                    mean = mean,
                    volatility = volatility,
                    persistence = persistence,
                    unconditionalProb = unconditionalProb
                )
            } else {
                // FIXED: Default parameters based on regime type
                val defaultParams = when (regime) {
                    0 -> RegimeParameters(0.0005, 0.015, 0.85, 0.4)   // Low vol: slight positive drift, low vol, high persistence
                    1 -> RegimeParameters(-0.001, 0.045, 0.65, 0.2)  // High vol: negative drift, high vol, medium persistence
                    2 -> RegimeParameters(0.002, 0.025, 0.75, 0.4)   // Trending: higher drift, medium vol, high persistence
                    else -> RegimeParameters(0.0, 0.02, 0.8, 0.33)
                }
                parameters[regime] = defaultParams
            }
        }

        return parameters
    }

    private fun calculateRegimePersistence(regimes: List<Int>, targetRegime: Int): Double {
        if (regimes.size < 2) return 0.8 // Default persistence

        var stayCount = 0
        var totalTransitions = 0

        for (i in 1 until regimes.size) {
            if (regimes[i - 1] == targetRegime) {
                totalTransitions++
                if (regimes[i] == targetRegime) {
                    stayCount++
                }
            }
        }

        return if (totalTransitions > 0) {
            stayCount.toDouble() / totalTransitions
        } else {
            0.8 // Default persistence
        }
    }

    // FIXED: Deterministic analytical forecasting instead of Monte Carlo
    private fun forecastAnalytical(
        currentPrice: Double,
        currentRegime: Int,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        transitionMatrix: Array<Array<Double>>,
        regimeParameters: Map<Int, RegimeParameters>
    ): Pair<Double, Double> {

        // 1. Calculate regime probabilities over time
        val regimeProbabilities = calculateRegimeProbabilitiesOverTime(
            currentRegime, daysAhead, transitionMatrix
        )

        // 2. Calculate price distribution for each regime
        val priceDistributions = calculatePriceDistributionsPerRegime(
            currentPrice, daysAhead, regimeParameters
        )

        // 3. Combine distributions weighted by regime probabilities
        val combinedDistribution = combineDistributions(regimeProbabilities, priceDistributions)

        // 4. Calculate barrier probabilities
        val upperProbability = calculateBarrierProbability(
            targetPrice = upperBand,
            currentPrice = currentPrice,
            distribution = combinedDistribution,
            daysAhead = daysAhead,
            isUpper = true
        )

        val lowerProbability = calculateBarrierProbability(
            targetPrice = lowerBand,
            currentPrice = currentPrice,
            distribution = combinedDistribution,
            daysAhead = daysAhead,
            isUpper = false
        )

        return Pair(upperProbability * 100, lowerProbability * 100)
    }

    private fun calculateRegimeProbabilitiesOverTime(
        currentRegime: Int,
        daysAhead: Int,
        transitionMatrix: Array<Array<Double>>
    ): Array<DoubleArray> {
        val numRegimes = config.numRegimes
        val regimeProbs = Array(daysAhead + 1) { DoubleArray(numRegimes) }

        // Initial state
        regimeProbs[0][currentRegime] = 1.0

        // Forward propagation using transition matrix
        for (day in 1..daysAhead) {
            for (toRegime in 0 until numRegimes) {
                var prob = 0.0
                for (fromRegime in 0 until numRegimes) {
                    prob += regimeProbs[day - 1][fromRegime] * transitionMatrix[fromRegime][toRegime]
                }
                regimeProbs[day][toRegime] = prob
            }
        }

        return regimeProbs
    }

    private fun calculatePriceDistributionsPerRegime(
        currentPrice: Double,
        daysAhead: Int,
        regimeParameters: Map<Int, RegimeParameters>
    ): Map<Int, Map<Double, Double>> {
        val distributions = mutableMapOf<Int, Map<Double, Double>>()

        regimeParameters.forEach { (regime, params) ->
            // Calculate price distribution for this regime
            val quantiles = listOf(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

            val distribution = quantiles.associateWith { q ->
                val zScore = getInverseNormal(q)
                val totalReturn = params.mean * daysAhead + zScore * params.volatility * sqrt(daysAhead.toDouble())
                currentPrice * exp(totalReturn)
            }

            distributions[regime] = distribution
        }

        return distributions
    }

    private fun combineDistributions(
        regimeProbabilities: Array<DoubleArray>,
        priceDistributions: Map<Int, Map<Double, Double>>
    ): Map<Double, Double> {
        val finalDay = regimeProbabilities.size - 1
        val quantiles = listOf(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

        return quantiles.associateWith { q ->
            var weightedPrice = 0.0

            priceDistributions.forEach { (regime, distribution) ->
                val regimeWeight = regimeProbabilities[finalDay][regime]
                val regimePrice = distribution[q] ?: 0.0
                weightedPrice += regimeWeight * regimePrice
            }

            weightedPrice
        }
    }

    private fun calculateBarrierProbability(
        targetPrice: Double,
        currentPrice: Double,
        distribution: Map<Double, Double>,
        daysAhead: Int,
        isUpper: Boolean
    ): Double {

        val sortedQuantiles = distribution.keys.sorted()
        val sortedPrices = sortedQuantiles.map { distribution[it]!! }

        // Get end-point probability through interpolation
        val endPointProb = interpolateProbability(targetPrice, sortedPrices, sortedQuantiles, isUpper)

        if (endPointProb <= 0.0 || endPointProb >= 1.0) return endPointProb

        // Apply barrier enhancement using reflection principle
        val logDistance = abs(ln(targetPrice / currentPrice))
        val impliedVolatility = estimateImpliedVolatility(distribution, currentPrice, daysAhead)
        val timeScaledVol = impliedVolatility * sqrt(daysAhead.toDouble())

        val d1 = logDistance / timeScaledVol

        val barrierProbability = if (d1 > 3.0) {
            endPointProb // Far barriers
        } else {
            minOf(1.0, 2.0 * endPointProb) // Near barriers with reflection principle
        }

        return barrierProbability
    }

    private fun estimateImpliedVolatility(
        distribution: Map<Double, Double>,
        currentPrice: Double,
        daysAhead: Int
    ): Double {
        val p90Price = distribution[0.9] ?: currentPrice
        val p10Price = distribution[0.1] ?: currentPrice

        if (p90Price <= p10Price) return 0.02 // Default volatility

        val priceRange = ln(p90Price / p10Price)
        return priceRange / (2.56 * sqrt(daysAhead.toDouble())) // 2.56 ≈ 2*Φ⁻¹(0.9)
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

    // Debug logging
    private fun logDebugInfo(
        timeSeries: List<TimeSeriesEntity>,
        regimes: List<Int>,
        transitionMatrix: Array<Array<Double>>,
        regimeParameters: Map<Int, RegimeParameters>,
        result: Pair<Double, Double>,
        daysAhead: Int
    ) {
        Log.d("RegimeSwitching", "=== REGIME SWITCHING DEBUG INFO ===")
        Log.d("RegimeSwitching", "Data points: ${timeSeries.size}")
        Log.d("RegimeSwitching", "Returns calculated: ${regimes.size}")
        Log.d("RegimeSwitching", "Current regime: ${regimes.lastOrNull()}")
        Log.d("RegimeSwitching", "Days ahead: $daysAhead")
        Log.d("RegimeSwitching", "Current price: ${timeSeries.last().price}")

        // Log regime distribution
        val regimeCounts = regimes.groupingBy { it }.eachCount()
        regimeCounts.forEach { (regime, count) ->
            val percentage = (count.toDouble() / regimes.size) * 100
            Log.d("RegimeSwitching", "Regime $regime: $count observations (${String.format("%.1f", percentage)}%)")
        }

        // Log transition matrix
        Log.d("RegimeSwitching", "Transition Matrix:")
        for (i in transitionMatrix.indices) {
            val row = transitionMatrix[i].joinToString(", ") { String.format("%.3f", it) }
            Log.d("RegimeSwitching", "  From $i: [$row]")
        }

        // Log regime parameters
        regimeParameters.forEach { (regime, params) ->
            Log.d("RegimeSwitching", "Regime $regime: mean=${String.format("%.6f", params.mean)}, " +
                    "vol=${String.format("%.4f", params.volatility)}, " +
                    "persist=${String.format("%.3f", params.persistence)}")
        }

        Log.d("RegimeSwitching", "Final probabilities: Upper=${String.format("%.1f", result.first)}%, " +
                "Lower=${String.format("%.1f", result.second)}%")
        Log.d("RegimeSwitching", "=== END DEBUG INFO ===")
    }
}