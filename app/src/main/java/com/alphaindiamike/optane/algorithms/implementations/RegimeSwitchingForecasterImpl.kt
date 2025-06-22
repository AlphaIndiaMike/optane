package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlin.math.*
import kotlin.random.Random

/**
 * RegimeSwitchingForecaster - Markov Chain Model
 * Regime-switching model with Markov chains
 */
class RegimeSwitchingForecasterImpl : AlgorithmRepository {

    // Regime parameters
    private data class RegimeParameters(
        val mean: Double,
        val volatility: Double,
        val probability: Double
    )

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 10) {
            return "upperBandProbability:0.0% ,lowerBandProbability:0.0% "
        }

        // 1. Calculate returns
        val returns = calculateReturns(timeSeries)

        // 2. Identify regimes
        val regimes = identifyRegimes(returns)

        // 3. Estimate transition matrix
        val transitionMatrix = estimateTransitionMatrix(regimes)

        // 4. Estimate regime parameters
        val regimeParameters = estimateRegimeParameters(returns, regimes)

        // 5. Forecast using Monte Carlo with regime switching
        val result = forecast(
            currentPrice = timeSeries.last().price,
            currentRegime = regimes.last(),
            upperBand = upperBand,
            lowerBand = lowerBand,
            daysAhead = daysAhead,
            transitionMatrix = transitionMatrix,
            regimeParameters = regimeParameters,
            numSimulations = 1000
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

    private fun identifyRegimes(returns: List<Double>): List<Int> {
        // Simple regime identification based on volatility clustering
        val regimes = mutableListOf<Int>()
        val window = 3

        for (i in returns.indices) {
            if (i < window) {
                regimes.add(0) // Default regime
                continue
            }

            val recentReturns = returns.subList(i - window, i)
            val recentVol = calculateVolatility(recentReturns)
            val recentReturn = abs(returns[i])

            // Classify regime: 0 = calm, 1 = volatile, 2 = trending
            val regime = when {
                recentVol < 0.02 -> 0 // Calm market
                recentReturn > 0.03 -> 1 // Volatile market
                else -> 2 // Trending market
            }
            regimes.add(regime)
        }

        return regimes
    }

    private fun calculateVolatility(returns: List<Double>): Double {
        if (returns.isEmpty()) return 0.0

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    private fun estimateTransitionMatrix(regimes: List<Int>): Array<Array<Double>> {
        val numRegimes = 3
        val transitions = Array(numRegimes) { Array(numRegimes) { 0.0 } }

        // Count transitions
        for (i in 1 until regimes.size) {
            val from = regimes[i - 1]
            val to = regimes[i]
            transitions[from][to]++
        }

        // Normalize to probabilities
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
        val numRegimes = 3
        val parameters = mutableMapOf<Int, RegimeParameters>()

        for (regime in 0 until numRegimes) {
            val regimeReturns = returns.filterIndexed { index, _ ->
                index < regimes.size && regimes[index] == regime
            }

            if (regimeReturns.isNotEmpty()) {
                val mean = regimeReturns.average()
                val variance = regimeReturns.map { (it - mean).pow(2) }.average()
                val volatility = sqrt(variance)
                val probability = regimeReturns.size.toDouble() / returns.size

                parameters[regime] = RegimeParameters(
                    mean = mean,
                    volatility = volatility,
                    probability = probability
                )
            } else {
                // Default parameters if no observations
                parameters[regime] = RegimeParameters(
                    mean = 0.0,
                    volatility = 0.02,
                    probability = 0.33
                )
            }
        }

        return parameters
    }

    private fun forecast(
        currentPrice: Double,
        currentRegime: Int,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        transitionMatrix: Array<Array<Double>>,
        regimeParameters: Map<Int, RegimeParameters>,
        numSimulations: Int
    ): Pair<Double, Double> {

        var reachesUpper = 0
        var reachesLower = 0

        repeat(numSimulations) {
            var regime = currentRegime
            var price = currentPrice
            var maxPrice = currentPrice
            var minPrice = currentPrice

            repeat(daysAhead) {
                // Simulate regime transition
                regime = simulateRegimeTransition(regime, transitionMatrix)

                // Generate return based on current regime
                val params = regimeParameters[regime] ?: RegimeParameters(0.0, 0.02, 0.33)
                val normalRand = boxMuller()
                val ret = params.mean + params.volatility * normalRand

                price *= exp(ret)
                maxPrice = max(maxPrice, price)
                minPrice = min(minPrice, price)
            }

            if (maxPrice >= upperBand) reachesUpper++
            if (minPrice <= lowerBand) reachesLower++
        }

        val probUpper = (reachesUpper.toDouble() / numSimulations) * 100
        val probLower = (reachesLower.toDouble() / numSimulations) * 100

        return Pair(probUpper, probLower)
    }

    private fun simulateRegimeTransition(currentRegime: Int, transitionMatrix: Array<Array<Double>>): Int {
        val rand = Random.nextDouble()
        var cumProb = 0.0

        for (nextRegime in 0 until 3) {
            cumProb += transitionMatrix[currentRegime][nextRegime]
            if (rand <= cumProb) {
                return nextRegime
            }
        }

        return currentRegime // Fallback
    }

    private fun boxMuller(): Double {
        // Box-Muller transformation for normal random numbers
        var u = 0.0
        var v = 0.0

        while (u == 0.0) u = Random.nextDouble() // Converting [0,1) to (0,1)
        while (v == 0.0) v = Random.nextDouble()

        return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
    }
}