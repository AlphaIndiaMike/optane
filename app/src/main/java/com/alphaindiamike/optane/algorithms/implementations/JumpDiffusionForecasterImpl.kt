package com.alphaindiamike.optane.algorithms.implementations

import kotlin.math.*
import kotlin.random.Random
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations

/**
 * Jump-Diffusion Model (Merton Model)
 * Captures sudden price movements with jump processes
 */
class JumpDiffusionForecasterImpl : AlgorithmRepository {

    // Jump-diffusion parameters
    private data class JumpParams(
        val jumpIntensity: Double,      // λ - jumps per year
        val averageJumpSize: Double,    // μ_J - average jump magnitude
        val jumpVolatility: Double,     // σ_J - jump size volatility
        val diffusionVolatility: Double // σ - continuous diffusion volatility
    )

    private data class JumpEvent(
        val index: Int,
        val jumpSize: Double,
        val isJump: Boolean
    )

    private data class MertonParameters(
        val drift: Double,              // μ - drift rate
        val diffusionVol: Double,       // σ - diffusion volatility
        val jumpIntensity: Double,      // λ - jump intensity
        val jumpMean: Double,           // μ_J - mean jump size
        val jumpStd: Double             // σ_J - jump size standard deviation
    )

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 10) {
            return "Insufficient data"
        }

        // 1. Calculate returns
        val returns = calculateReturns(timeSeries)

        // 2. Detect jumps in the return series
        val jumpEvents = detectJumps(returns, threshold = 3.0)

        // 3. Estimate jump-diffusion parameters
        val jumpParams = estimateJumpParameters(returns, jumpEvents)

        // 4. Separate diffusion component
        val diffusionReturns = separateDiffusionComponent(returns, jumpEvents)

        // 5. Run Monte Carlo simulation with jumps
        val result = monteCarloWithJumps(
            currentPrice = timeSeries.last().price,
            upperBand = upperBand,
            lowerBand = lowerBand,
            daysAhead = daysAhead,
            jumpParams = jumpParams,
            numSimulations = 10000
        )

        return """
            Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", result.first)}%
            Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", result.second)}%
            """.trimIndent()
    }

    private fun calculateReturns(timeSeries: List<TimeSeriesEntity>): List<Double> {
        val returns = mutableListOf<Double>()
        for (i in 1 until timeSeries.size) {
            val ret = ln(timeSeries[i].price / timeSeries[i-1].price)
            returns.add(ret)
        }
        return returns
    }

    private fun detectJumps(returns: List<Double>, threshold: Double = 3.0): List<JumpEvent> {
        if (returns.size < 5) return emptyList()

        // Calculate rolling statistics for jump detection
        val window = minOf(10, returns.size / 2)
        val jumpEvents = mutableListOf<JumpEvent>()

        for (i in window until returns.size - window) {
            // Calculate local statistics
            val localReturns = returns.subList(i - window, i + window)
            val localMean = localReturns.average()
            val localStd = sqrt(localReturns.map { (it - localMean).pow(2) }.average())

            // Detect jump using threshold method
            val standardizedReturn = abs(returns[i] - localMean) / (localStd + 1e-8)
            val isJump = standardizedReturn > threshold

            jumpEvents.add(JumpEvent(
                index = i,
                jumpSize = if (isJump) returns[i] - localMean else 0.0,
                isJump = isJump
            ))
        }

        return jumpEvents
    }

    private fun estimateJumpParameters(returns: List<Double>, jumps: List<JumpEvent>): JumpParams {
        // Separate jump and non-jump returns
        val jumpReturns = jumps.filter { it.isJump }.map { it.jumpSize }
        val nonJumpReturns = returns.filterIndexed { index, _ ->
            jumps.find { it.index == index }?.isJump != true
        }

        // Estimate jump parameters
        val jumpIntensity = if (returns.isNotEmpty()) {
            (jumpReturns.size.toDouble() / returns.size) * 252.0 // Annualized
        } else 0.0

        val averageJumpSize = if (jumpReturns.isNotEmpty()) {
            jumpReturns.average()
        } else 0.0

        val jumpVolatility = if (jumpReturns.size > 1) {
            val jumpMean = jumpReturns.average()
            sqrt(jumpReturns.map { (it - jumpMean).pow(2) }.average())
        } else 0.02

        // Estimate diffusion volatility from non-jump returns
        val diffusionVolatility = if (nonJumpReturns.size > 1) {
            val mean = nonJumpReturns.average()
            sqrt(nonJumpReturns.map { (it - mean).pow(2) }.average())
        } else 0.02

        return JumpParams(
            jumpIntensity = jumpIntensity,
            averageJumpSize = averageJumpSize,
            jumpVolatility = jumpVolatility,
            diffusionVolatility = diffusionVolatility
        )
    }

    private fun separateDiffusionComponent(returns: List<Double>, jumpEvents: List<JumpEvent>): List<Double> {
        // Remove jump components to isolate diffusion
        val diffusionReturns = returns.toMutableList()

        jumpEvents.forEach { jumpEvent ->
            if (jumpEvent.isJump && jumpEvent.index < diffusionReturns.size) {
                diffusionReturns[jumpEvent.index] = diffusionReturns[jumpEvent.index] - jumpEvent.jumpSize
            }
        }

        return diffusionReturns
    }

    private fun monteCarloWithJumps(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        jumpParams: JumpParams,
        numSimulations: Int
    ): Pair<Double, Double> {

        var reachesUpper = 0
        var reachesLower = 0

        repeat(numSimulations) {
            val path = simulateJumpDiffusionPath(
                startPrice = currentPrice,
                timeHorizon = daysAhead,
                jumpParams = jumpParams
            )

            val maxPrice = path.maxOrNull() ?: currentPrice
            val minPrice = path.minOrNull() ?: currentPrice

            if (maxPrice >= upperBand) reachesUpper++
            if (minPrice <= lowerBand) reachesLower++
        }

        val probUpper = (reachesUpper.toDouble() / numSimulations) * 100
        val probLower = (reachesLower.toDouble() / numSimulations) * 100

        return Pair(probUpper, probLower)
    }

    private fun simulateJumpDiffusionPath(
        startPrice: Double,
        timeHorizon: Int,
        jumpParams: JumpParams
    ): List<Double> {

        val dt = 1.0 / 252.0 // Daily time step
        val prices = mutableListOf<Double>()
        var currentPrice = startPrice

        prices.add(currentPrice)

        repeat(timeHorizon) {
            // 1. Generate number of jumps in this time step (Poisson process)
            val numJumps = poissonProcess(jumpParams.jumpIntensity * dt)

            // 2. Generate diffusion component (Geometric Brownian Motion)
            val normalRandom = generateNormalRandom()
            val diffusionComponent = exp(
                (0.0 - 0.5 * jumpParams.diffusionVolatility.pow(2)) * dt +
                        jumpParams.diffusionVolatility * sqrt(dt) * normalRandom
            )

            // 3. Generate jump component
            var jumpComponent = 1.0
            repeat(numJumps) {
                val jumpSize = jumpParams.averageJumpSize +
                        jumpParams.jumpVolatility * generateNormalRandom()
                jumpComponent *= exp(jumpSize)
            }

            // 4. Combined price evolution: S(t+dt) = S(t) * diffusion * jump
            currentPrice *= diffusionComponent * jumpComponent
            prices.add(currentPrice)
        }

        return prices
    }

    private fun poissonProcess(lambda: Double): Int {
        // Generate Poisson-distributed number of jumps
        if (lambda <= 0.0) return 0

        val L = exp(-lambda)
        var k = 0
        var p = 1.0

        do {
            k++
            p *= Random.nextDouble()
        } while (p > L)

        return k - 1
    }

    private fun generateNormalRandom(): Double {
        // Box-Muller transformation
        var u = 0.0
        var v = 0.0

        while (u == 0.0) u = Random.nextDouble()
        while (v == 0.0) v = Random.nextDouble()

        return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
    }

    // Advanced jump detection using Lee-Mykland test
    private fun leeMylandJumpTest(returns: List<Double>, alpha: Double = 0.01): List<JumpEvent> {
        if (returns.size < 20) return emptyList()

        val jumpEvents = mutableListOf<JumpEvent>()
        val window = 20

        for (i in window until returns.size) {
            // Calculate realized volatility using recent returns
            val recentReturns = returns.subList(i - window, i)
            val realizedVar = recentReturns.map { it.pow(2) }.sum()
            val realizedVol = sqrt(realizedVar)

            // Lee-Mykland test statistic
            val testStat = abs(returns[i]) / realizedVol
            val criticalValue = sqrt(2 * ln(1 / alpha))

            val isJump = testStat > criticalValue

            jumpEvents.add(JumpEvent(
                index = i,
                jumpSize = if (isJump) returns[i] else 0.0,
                isJump = isJump
            ))
        }

        return jumpEvents
    }

    // Calibrate Merton model using Maximum Likelihood Estimation (simplified)
    private fun calibrateMertonModel(returns: List<Double>): MertonParameters {
        val jumps = detectJumps(returns, 2.5)
        val jumpReturns = jumps.filter { it.isJump }.map { it.jumpSize }

        // Estimate parameters using method of moments (simplified MLE)
        val allReturnsMean = returns.average()
        val allReturnsVar = returns.map { (it - allReturnsMean).pow(2) }.average()

        val jumpFreq = jumpReturns.size.toDouble() / returns.size
        val annualJumpIntensity = jumpFreq * 252.0

        val jumpMean = if (jumpReturns.isNotEmpty()) jumpReturns.average() else 0.0
        val jumpVar = if (jumpReturns.size > 1) {
            jumpReturns.map { (it - jumpMean).pow(2) }.average()
        } else 0.01

        // Adjust for jump contribution to total variance
        val jumpContribution = annualJumpIntensity * (jumpMean.pow(2) + jumpVar)
        val diffusionVar = max(0.01, allReturnsVar - jumpContribution / 252.0)

        return MertonParameters(
            drift = allReturnsMean * 252.0, // Annualized
            diffusionVol = sqrt(diffusionVar * 252.0), // Annualized
            jumpIntensity = annualJumpIntensity,
            jumpMean = jumpMean,
            jumpStd = sqrt(jumpVar)
        )
    }

    // Calculate option prices using Merton jump-diffusion formula
    private fun mertonOptionPrice(
        spot: Double,
        strike: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        params: MertonParameters,
        isCall: Boolean = true
    ): Double {

        val maxTerms = 10 // Truncate infinite series
        var optionPrice = 0.0

        for (n in 0 until maxTerms) {
            // Poisson probability
            val poissonProb = exp(-params.jumpIntensity * timeToExpiry) *
                    (params.jumpIntensity * timeToExpiry).pow(n) / factorial(n)

            // Adjusted parameters for n jumps
            val adjustedVol = sqrt(params.diffusionVol.pow(2) + n * params.jumpStd.pow(2) / timeToExpiry)
            val adjustedRate = riskFreeRate - params.jumpIntensity *
                    (exp(params.jumpMean + 0.5 * params.jumpStd.pow(2)) - 1) +
                    n * ln(1 + params.jumpMean) / timeToExpiry

            // Black-Scholes price with adjusted parameters
            val bsPrice = blackScholesPrice(spot, strike, timeToExpiry, adjustedRate, adjustedVol, isCall)

            optionPrice += poissonProb * bsPrice
        }

        return optionPrice
    }

    private fun factorial(n: Int): Double {
        return if (n <= 1) 1.0 else n * factorial(n - 1)
    }

    private fun blackScholesPrice(
        spot: Double,
        strike: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double,
        isCall: Boolean
    ): Double {

        val d1 = (ln(spot / strike) + (riskFreeRate + 0.5 * volatility.pow(2)) * timeToExpiry) /
                (volatility * sqrt(timeToExpiry))
        val d2 = d1 - volatility * sqrt(timeToExpiry)

        return if (isCall) {
            spot * normalCDF(d1) - strike * exp(-riskFreeRate * timeToExpiry) * normalCDF(d2)
        } else {
            strike * exp(-riskFreeRate * timeToExpiry) * normalCDF(-d2) - spot * normalCDF(-d1)
        }
    }

    private fun normalCDF(x: Double): Double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    }

    private fun erf(x: Double): Double {
        // Error function approximation
        val a1 = 0.254829592
        val a2 = -0.284496736
        val a3 = 1.421413741
        val a4 = -1.453152027
        val a5 = 1.061405429
        val p = 0.3275911

        val sign = if (x < 0) -1 else 1
        val absX = abs(x)

        val t = 1.0 / (1.0 + p * absX)
        val y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absX * absX)

        return sign * y
    }

    // Jump size distribution analysis
    private fun analyzeJumpDistribution(jumpEvents: List<JumpEvent>): Map<String, Double> {
        val jumpSizes = jumpEvents.filter { it.isJump }.map { it.jumpSize }

        if (jumpSizes.isEmpty()) {
            return mapOf(
                "jumpFrequency" to 0.0,
                "averageJumpSize" to 0.0,
                "jumpVolatility" to 0.0,
                "positiveJumps" to 0.0,
                "negativeJumps" to 0.0
            )
        }

        val positiveJumps = jumpSizes.filter { it > 0 }
        val negativeJumps = jumpSizes.filter { it < 0 }

        return mapOf(
            "jumpFrequency" to jumpSizes.size.toDouble(),
            "averageJumpSize" to jumpSizes.average(),
            "jumpVolatility" to sqrt(jumpSizes.map { (it - jumpSizes.average()).pow(2) }.average()),
            "positiveJumps" to positiveJumps.size.toDouble(),
            "negativeJumps" to negativeJumps.size.toDouble(),
            "averagePositiveJump" to if (positiveJumps.isNotEmpty()) positiveJumps.average() else 0.0,
            "averageNegativeJump" to if (negativeJumps.isNotEmpty()) negativeJumps.average() else 0.0
        )
    }
}