package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import kotlin.random.Random

import android.util.Log

/**
 * Monte Carlo Basic Implementation
 * Simple Monte Carlo simulation with geometric Brownian motion
 */
class MonteCarloBasicImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

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

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        // 1. Calculate historical parameters
        val returns = calculateReturns(timeSeries)
        val meanReturn = calculateMeanReturn(returns)
        val volatility = calculateHistoricalVolatility(returns)
        val currentPrice = timeSeries.last().price

        // 2. Calculate drift - scale to daily
        val riskFreeRate = 0.025 // ECB rate approximation
        val drift = meanReturn // Use historical mean return (already daily)

        // Debug logging
        if (enableDebugLogging) {
            logDebugInfo(currentPrice, meanReturn, volatility, drift, daysAhead)
        }

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
        val finalPrices = mutableListOf<Double>()

        repeat(numSimulations) {
            val path = geometricBrownianMotion(
                startPrice = currentPrice,
                drift = drift,
                volatility = volatility,
                timeSteps = daysAhead,
                totalTime = daysAhead.toDouble() // Use days directly, not scaled to years
            )

            // FIXED: Check if price touches/reaches bands at ANY point during the period
            if (path.any { it >= upperBand }) reachesUpper++
            if (path.any { it <= lowerBand }) reachesLower++

            // Store final price for statistics
            finalPrices.add(path.last())

            // Debug first few simulations
            if (enableDebugLogging && finalPrices.size <= 5) {
                val touchesUpper = path.any { it >= upperBand }
                val touchesLower = path.any { it <= lowerBand }
                val maxPrice = path.maxOrNull() ?: currentPrice
                val minPrice = path.minOrNull() ?: currentPrice
                Log.d("MonteCarloBasic", "Simulation ${finalPrices.size}: Max=$maxPrice, Min=$minPrice, TouchesUpper=$touchesUpper, TouchesLower=$touchesLower")
            }
        }

        val probUpper = (reachesUpper.toDouble() / numSimulations) * 100
        val probLower = (reachesLower.toDouble() / numSimulations) * 100

        // Debug logging for simulation results
        if (enableDebugLogging) {
            val stats = calculateStatistics(finalPrices)
            val confidence = getConfidenceIntervals(finalPrices)
            Log.d("MonteCarloBasic", "=== Simulation Results ===")
            Log.d("MonteCarloBasic", "Target Bands: Lower=$lowerBand, Upper=$upperBand")
            Log.d("MonteCarloBasic", "Final Price Statistics: $stats")
            Log.d("MonteCarloBasic", "95% Confidence Interval: $confidence")
            Log.d("MonteCarloBasic", "Upper Band Hits: $reachesUpper / $numSimulations")
            Log.d("MonteCarloBasic", "Lower Band Hits: $reachesLower / $numSimulations")
            Log.d("MonteCarloBasic", "Sample final prices: ${finalPrices.take(10)}")
        }

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

            // FIXED: GBM formula without double drift adjustment
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

    // Debug logging method
    private fun logDebugInfo(currentPrice: Double, meanReturn: Double, volatility: Double, drift: Double, daysAhead: Int) {
        Log.d("MonteCarloBasic", "=== Monte Carlo Parameters ===")
        Log.d("MonteCarloBasic", "Current Price: $currentPrice")
        Log.d("MonteCarloBasic", "Historical Mean Return: $meanReturn")
        Log.d("MonteCarloBasic", "Daily Volatility: $volatility")
        Log.d("MonteCarloBasic", "Annual Volatility: ${volatility * sqrt(252.0)}")
        Log.d("MonteCarloBasic", "Risk-Free Rate (Drift): $drift")
        Log.d("MonteCarloBasic", "Days Ahead: $daysAhead")
        Log.d("MonteCarloBasic", "Number of Simulations: ${MCConfig().numSimulations}")
    }

    // Additional utility methods for confidence intervals and statistics - now integrated for debugging
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