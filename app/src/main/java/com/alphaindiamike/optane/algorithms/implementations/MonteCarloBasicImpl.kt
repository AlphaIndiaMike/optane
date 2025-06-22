package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import kotlin.random.Random

/**
 * Monte Carlo Basic Implementation
 * Simple Monte Carlo simulation with geometric Brownian motion
 */
class MonteCarloBasicImpl : AlgorithmRepository {

    // Monte Carlo configuration
    private data class MCConfig(
        val numSimulations: Int = 10000,
        val timeSteps: Int = 252,  // Daily steps
        val confidenceLevel: Double = 0.95
    )

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 5) {
            return "upperBandProbability:0.0,lowerBandProbability:0.0"
        }

        // 1. Calculate historical parameters
        val returns = calculateReturns(timeSeries)
        val meanReturn = calculateMeanReturn(returns)
        val volatility = calculateHistoricalVolatility(returns)
        val currentPrice = timeSeries.last().price

        // 2. Calculate drift (risk-neutral measure)
        val riskFreeRate = 0.025 // ECB rate approximation
        val drift = riskFreeRate - 0.5 * volatility.pow(2)

        // 3. Run Monte Carlo simulation
        val result = runMonteCarloSimulation(
            currentPrice = currentPrice,
            drift = drift,
            volatility = volatility,
            upperBand = upperBand,
            lowerBand = lowerBand,
            daysAhead = daysAhead,
            numSimulations = MCConfig().numSimulations
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

    private fun calculateMeanReturn(returns: List<Double>): Double {
        return if (returns.isNotEmpty()) returns.average() else 0.0
    }

    private fun calculateHistoricalVolatility(returns: List<Double>): Double {
        if (returns.size < 2) return 0.02 // Default 2% daily volatility

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.sum() / (returns.size - 1)
        return sqrt(variance)
    }

    private fun runMonteCarloSimulation(
        currentPrice: Double,
        drift: Double,
        volatility: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        numSimulations: Int
    ): Pair<Double, Double> {

        var reachesUpper = 0
        var reachesLower = 0

        repeat(numSimulations) {
            val path = geometricBrownianMotion(
                startPrice = currentPrice,
                drift = drift,
                volatility = volatility,
                timeSteps = daysAhead,
                totalTime = daysAhead / 252.0 // Convert days to years
            )

            // Check if path hits target bands
            val maxPrice = path.maxOrNull() ?: currentPrice
            val minPrice = path.minOrNull() ?: currentPrice

            if (maxPrice >= upperBand) reachesUpper++
            if (minPrice <= lowerBand) reachesLower++
        }

        val probUpper = (reachesUpper.toDouble() / numSimulations) * 100
        val probLower = (reachesLower.toDouble() / numSimulations) * 100

        return Pair(probUpper, probLower)
    }

    private fun geometricBrownianMotion(
        startPrice: Double,
        drift: Double,
        volatility: Double,
        timeSteps: Int,
        totalTime: Double
    ): List<Double> {
        val dt = totalTime / timeSteps
        val prices = mutableListOf<Double>()
        var currentPrice = startPrice

        prices.add(currentPrice)

        repeat(timeSteps) {
            val normalRand = generateNormalRandom()

            // GBM formula: dS = S * (μ * dt + σ * sqrt(dt) * dW)
            val priceChange = exp((drift - 0.5 * volatility.pow(2)) * dt + volatility * sqrt(dt) * normalRand)
            currentPrice *= priceChange

            prices.add(currentPrice)
        }

        return prices
    }

    private fun generateNormalRandom(): Double {
        // Box-Muller transformation for normal random numbers
        return boxMuller()
    }

    private fun boxMuller(): Double {
        // Box-Muller transformation
        var u = 0.0
        var v = 0.0

        while (u == 0.0) u = Random.nextDouble() // Converting [0,1) to (0,1)
        while (v == 0.0) v = Random.nextDouble()

        return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
    }

    // Additional utility methods for confidence intervals and statistics

    private fun getConfidenceIntervals(results: List<Double>): Pair<Double, Double> {
        if (results.isEmpty()) return Pair(0.0, 0.0)

        val sorted = results.sorted()
        val n = sorted.size
        val lowerIndex = (n * 0.025).toInt() // 2.5% percentile
        val upperIndex = (n * 0.975).toInt() // 97.5% percentile

        return Pair(
            sorted.getOrElse(lowerIndex) { sorted.first() },
            sorted.getOrElse(upperIndex) { sorted.last() }
        )
    }

    private fun calculateStatistics(results: List<Double>): Map<String, Double> {
        if (results.isEmpty()) return emptyMap()

        val sorted = results.sorted()
        val n = results.size

        return mapOf(
            "mean" to results.average(),
            "median" to sorted[n / 2],
            "std" to sqrt(results.map { (it - results.average()).pow(2) }.average()),
            "min" to sorted.first(),
            "max" to sorted.last(),
            "p5" to sorted[(n * 0.05).toInt()],
            "p95" to sorted[(n * 0.95).toInt()]
        )
    }
}