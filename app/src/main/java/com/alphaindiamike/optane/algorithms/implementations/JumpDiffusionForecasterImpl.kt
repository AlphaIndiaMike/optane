package com.alphaindiamike.optane.algorithms.implementations

import kotlin.math.*
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations
import android.util.Log
import kotlin.random.Random
/**
 * Jump-Diffusion Forecaster - Final Implementation
 * Calculates probability of touching price bands using first-passage time formulas
 */
class JumpDiffusionForecasterImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

    private data class JumpDiffusionParameters(
        val diffusionMean: Double,
        val diffusionVolatility: Double,
        val jumpIntensity: Double,
        val jumpMean: Double,
        val jumpVolatility: Double
    )

    private data class JumpEvent(
        val index: Int,
        val jumpSize: Double,
        val isJump: Boolean,
        val confidence: Double
    )

    private data class JumpDiffusionConfig(
        val minObservations: Int = 30,
        val jumpDetectionWindow: Int = 10,
        val confidenceLevel: Double = 0.99,
        val maxHistoryDays: Int = 252
    )

    private val config = JumpDiffusionConfig()

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < config.minObservations) {
            return "Insufficient data"
        }

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        try {
            val limitedTimeSeries = if (timeSeries.size > config.maxHistoryDays) {
                timeSeries.takeLast(config.maxHistoryDays)
            } else {
                timeSeries
            }

            // 1. Calculate returns
            val returns = calculateReturns(limitedTimeSeries)

            // 2. Detect jumps
            val jumpEvents = detectJumpsDeterministic(returns)

            // 3. Estimate parameters
            val parameters = estimateJumpDiffusionParameters(returns, jumpEvents)

            // 4. Calculate first-passage probabilities
            val currentPrice = limitedTimeSeries.last().price
            val upperProbability = calculateFirstPassageProbability(
                currentPrice = currentPrice,
                barrier = upperBand,
                timeHorizon = daysAhead.toDouble(),
                parameters = parameters,
                isUpper = true
            )

            val lowerProbability = calculateFirstPassageProbability(
                currentPrice = currentPrice,
                barrier = lowerBand,
                timeHorizon = daysAhead.toDouble(),
                parameters = parameters,
                isUpper = false
            )

            if (enableDebugLogging) {
                logDebugInfo(currentPrice, upperBand, lowerBand, daysAhead, parameters, upperProbability, lowerProbability)
            }

            return """
                Upper band of ${upperBand} probability: ${String.format("%.1f", upperProbability * 100)}%
                Lower band of ${lowerBand} probability: ${String.format("%.1f", lowerProbability * 100)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("JumpDiffusion", "Error in calculation: ${e.message}")
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

    private fun detectJumpsDeterministic(returns: List<Double>): List<JumpEvent> {
        if (returns.size < config.jumpDetectionWindow * 2) {
            return List(returns.size) { JumpEvent(it, 0.0, false, 0.0) }
        }

        val jumpEvents = mutableListOf<JumpEvent>()
        val window = config.jumpDetectionWindow
        val globalMean = returns.average()
        val globalStd = sqrt(returns.map { (it - globalMean).pow(2) }.average())

        for (i in window until returns.size - window) {
            val beforeWindow = returns.subList(maxOf(0, i - window), i)
            val afterWindow = returns.subList(i + 1, minOf(returns.size, i + window + 1))
            val localReturns = beforeWindow + afterWindow

            if (localReturns.isNotEmpty()) {
                val localMean = localReturns.average()
                val localStd = sqrt(localReturns.map { (it - localMean).pow(2) }.average())
                val adaptiveStd = maxOf(localStd, globalStd * 0.5)
                val threshold = 4.0

                val standardizedReturn = abs(returns[i] - localMean) / adaptiveStd
                val isJump = standardizedReturn > threshold
                val confidence = if (isJump) min(0.99, standardizedReturn / threshold) else 0.0

                jumpEvents.add(JumpEvent(
                    index = i,
                    jumpSize = if (isJump) returns[i] - localMean else 0.0,
                    isJump = isJump,
                    confidence = confidence
                ))
            } else {
                jumpEvents.add(JumpEvent(i, 0.0, false, 0.0))
            }
        }

        return jumpEvents
    }

    private fun estimateJumpDiffusionParameters(
        returns: List<Double>,
        jumpEvents: List<JumpEvent>
    ): JumpDiffusionParameters {

        val jumps = jumpEvents.filter { it.isJump }
        val jumpSizes = jumps.map { it.jumpSize }

        // Calculate diffusion component (returns without jumps)
        val diffusionReturns = returns.filterIndexed { index, _ ->
            jumpEvents.find { it.index == index }?.isJump != true
        }

        val diffusionMean = if (diffusionReturns.isNotEmpty()) {
            diffusionReturns.average()
        } else {
            returns.average()
        }

        val diffusionVolatility = if (diffusionReturns.size > 1) {
            val mean = diffusionReturns.average()
            sqrt(diffusionReturns.map { (it - mean).pow(2) }.average())
        } else {
            sqrt(returns.map { (it - returns.average()).pow(2) }.average())
        }

        // Jump parameters (daily units)
        val jumpIntensity = if (returns.isNotEmpty()) {
            jumps.size.toDouble() / returns.size
        } else {
            0.0
        }

        val jumpMean = if (jumpSizes.isNotEmpty()) {
            jumpSizes.average()
        } else {
            0.0
        }

        val jumpVolatility = if (jumpSizes.size > 1) {
            val mean = jumpSizes.average()
            sqrt(jumpSizes.map { (it - mean).pow(2) }.average())
        } else {
            maxOf(0.01, diffusionVolatility)
        }

        // Smart volatility constraint: only boost if it's unrealistically low AND we have normal market data
        val minVolatility = if (diffusionVolatility < 0.001 && jumps.size < returns.size * 0.05) {
            // Very low vol + few jumps = probably constant data, keep it low
            0.001
        } else {
            // Normal market data with some volatility
            0.015
        }

        return JumpDiffusionParameters(
            diffusionMean = maxOf(-0.2, minOf(0.2, diffusionMean)),
            diffusionVolatility = maxOf(minVolatility, minOf(0.2, diffusionVolatility)),
            jumpIntensity = maxOf(0.0, minOf(0.2, jumpIntensity)),
            jumpMean = maxOf(-0.5, minOf(0.5, jumpMean)),
            jumpVolatility = maxOf(0.01, minOf(0.5, jumpVolatility))
        )
    }

    private fun calculateFirstPassageProbability(
        currentPrice: Double,
        barrier: Double,
        timeHorizon: Double,
        parameters: JumpDiffusionParameters,
        isUpper: Boolean
    ): Double {

        // Check if already past barrier
        if ((isUpper && currentPrice >= barrier) || (!isUpper && currentPrice <= barrier)) {
            return 1.0
        }

        val maxTerms = 10
        var totalProbability = 0.0

        // Sum over possible number of jumps
        for (n in 0 until maxTerms) {
            val lambda = parameters.jumpIntensity * timeHorizon
            val poissonProb = if (lambda > 0) {
                exp(-lambda) * lambda.pow(n) / factorial(n)
            } else {
                if (n == 0) 1.0 else 0.0
            }

            val conditionalProb = calculateConditionalFirstPassage(
                currentPrice = currentPrice,
                barrier = barrier,
                timeHorizon = timeHorizon,
                numJumps = n,
                parameters = parameters,
                isUpper = isUpper
            )

            totalProbability += poissonProb * conditionalProb

            if (poissonProb < 1e-6) break
        }

        return minOf(1.0, maxOf(0.0, totalProbability))
    }

    private fun calculateConditionalFirstPassage(
        currentPrice: Double,
        barrier: Double,
        timeHorizon: Double,
        numJumps: Int,
        parameters: JumpDiffusionParameters,
        isUpper: Boolean
    ): Double {

        // Include jump effects
        val totalJumpContribution = numJumps * parameters.jumpMean
        val totalJumpVariance = numJumps * parameters.jumpVolatility.pow(2)

        val effectiveDrift = parameters.diffusionMean
        val effectiveVolatility = sqrt(
            parameters.diffusionVolatility.pow(2) + totalJumpVariance / timeHorizon
        )

        // Adjust starting price by jumps
        val adjustedStartPrice = currentPrice * exp(totalJumpContribution)

        return calculateFirstPassageGBM(
            startPrice = adjustedStartPrice,
            barrier = barrier,
            drift = effectiveDrift,
            volatility = effectiveVolatility,
            timeHorizon = timeHorizon,
            isUpper = isUpper
        )
    }

    private fun generateNormalRandom(): Double {
        val u1 = kotlin.random.Random.Default.nextDouble()
        val u2 = kotlin.random.Random.Default.nextDouble()
        return sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    /**
     * Correct first-passage probability formula for GBM
     */
    /*
    private fun calculateFirstPassageGBM(
        startPrice: Double,
        barrier: Double,
        drift: Double,
        volatility: Double,
        timeHorizon: Double,
        isUpper: Boolean
    ): Double {

        // Handle edge case: already at or past barrier
        if ((isUpper && startPrice >= barrier) || (!isUpper && startPrice <= barrier)) {
            return 1.0
        }

        if (timeHorizon <= 0.0 || volatility <= 0.0) {
            return 0.0
        }

        val mu = drift
        val sigma = volatility
        val T = timeHorizon
        val S0 = startPrice
        val B = barrier

        val logRatio = ln(B / S0)
        val sigmaSqrtT = sigma * sqrt(T)

        // First-passage probability using correct analytical formula
        val d1 = (-logRatio + mu * T) / sigmaSqrtT
        val d2 = (-logRatio - mu * T) / sigmaSqrtT

        val prob1 = if (isUpper) {
            1.0 - normalCDF(d1)
        } else {
            normalCDF(d1)
        }

        val alpha = 2.0 * mu / (sigma * sigma)
        val reflectionTerm = (B / S0).pow(alpha)

        val prob2 = if (isUpper) {
            reflectionTerm * normalCDF(d2)
        } else {
            reflectionTerm * (1.0 - normalCDF(d2))
        }

        return minOf(1.0, maxOf(0.0, prob1 + prob2))
    }*/

    private fun calculateFirstPassageGBM(
        startPrice: Double,
        barrier: Double,
        drift: Double,
        volatility: Double,
        timeHorizon: Double,
        isUpper: Boolean
    ): Double {

        // Check if already past barrier
        if ((isUpper && startPrice >= barrier) || (!isUpper && startPrice <= barrier)) {
            return 1.0
        }

        val numSimulations = 5000
        var touchCount = 0
        val numSteps = timeHorizon.toInt()

        repeat(numSimulations) {
            var currentPrice = startPrice
            var touched = false

            for (step in 1..numSteps) {
                // Generate random normal
                val random = generateNormalRandom()

                // Update price using GBM with jump-diffusion parameters
                val priceChange = drift + volatility * random
                currentPrice *= exp(priceChange)

                // Check if barrier touched
                if ((isUpper && currentPrice >= barrier) || (!isUpper && currentPrice <= barrier)) {
                    touched = true
                    break
                }
            }

            if (touched) touchCount++
        }

        return touchCount.toDouble() / numSimulations
    }

    private fun normalCDF(x: Double): Double {
        if (x < -6.0) return 0.0
        if (x > 6.0) return 1.0

        val b1 = 0.319381530
        val b2 = -0.356563782
        val b3 = 1.781477937
        val b4 = -1.821255978
        val b5 = 1.330274429
        val p = 0.2316419
        val c = 0.39894228

        val absX = abs(x)
        val t = 1.0 / (1.0 + p * absX)
        val phi = c * exp(-0.5 * absX * absX) * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))

        return if (x >= 0.0) 1.0 - phi else phi
    }

    private fun factorial(n: Int): Double {
        if (n <= 1) return 1.0
        var result = 1.0
        for (i in 2..n) {
            result *= i
        }
        return result
    }

    private fun logDebugInfo(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        parameters: JumpDiffusionParameters,
        upperProb: Double,
        lowerProb: Double
    ) {
        Log.d("JumpDiffusion", "=== FIRST-PASSAGE PROBABILITY CALCULATION ===")
        Log.d("JumpDiffusion", "Current price: $currentPrice")
        Log.d("JumpDiffusion", "Upper band: $upperBand, Lower band: $lowerBand")
        Log.d("JumpDiffusion", "Days ahead: $daysAhead")
        Log.d("JumpDiffusion", "Parameters:")
        Log.d("JumpDiffusion", "  Daily drift: ${String.format("%.6f", parameters.diffusionMean)}")
        Log.d("JumpDiffusion", "  Daily volatility: ${String.format("%.4f", parameters.diffusionVolatility)}")
        Log.d("JumpDiffusion", "  Daily jump intensity: ${String.format("%.4f", parameters.jumpIntensity)}")
        Log.d("JumpDiffusion", "  Jump mean: ${String.format("%.4f", parameters.jumpMean)}")
        Log.d("JumpDiffusion", "  Jump volatility: ${String.format("%.4f", parameters.jumpVolatility)}")
        Log.d("JumpDiffusion", "Results:")
        Log.d("JumpDiffusion", "  Upper probability: ${String.format("%.1f", upperProb * 100)}%")
        Log.d("JumpDiffusion", "  Lower probability: ${String.format("%.1f", lowerProb * 100)}%")
        Log.d("JumpDiffusion", "=== END DEBUG INFO ===")
    }
}