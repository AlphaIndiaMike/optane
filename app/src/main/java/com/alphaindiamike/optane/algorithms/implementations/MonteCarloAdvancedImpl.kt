package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import kotlin.math.*
import kotlin.random.Random
import android.util.Log

/**
 * Monte Carlo Advanced Implementation
 * Advanced Monte Carlo with variance reduction techniques
 */
class MonteCarloAdvancedImpl(private val enableDebugLogging: Boolean = false) : AlgorithmRepository {

    // Advanced Monte Carlo configuration
    private data class AdvancedMCConfig(
        val numSimulations: Int = 50000,
        val useAntitheticVariates: Boolean = true,
        val useControlVariates: Boolean = true,
        val useStratifiedSampling: Boolean = true,
        val numStrata: Int = 10
    )

    private data class Greeks(
        val delta: Double,
        val gamma: Double,
        val vega: Double,
        val theta: Double
    )

    private data class VaRResult(
        val var95: Double,  // 95% Value at Risk
        val var99: Double,  // 99% Value at Risk
        val expectedShortfall: Double,
        val maxDrawdown: Double
    )

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 10) {
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

        // 2. Calculate drift - FIXED: use historical mean return for daily data
        val riskFreeRate = 0.025 / 252.0 // Convert annual to daily
        val drift = meanReturn // Use historical mean return (already daily)

        // Debug logging
        if (enableDebugLogging) {
            logDebugInfo(currentPrice, meanReturn, volatility, drift, daysAhead)
        }

        // 3. Run advanced Monte Carlo simulation
        val config = AdvancedMCConfig()
        val (probUpper, probLower, simulationResults) = runAdvancedMonteCarloSimulation(
            currentPrice = currentPrice,
            drift = drift,
            volatility = volatility,
            upperBand = upperBand,
            lowerBand = lowerBand,
            daysAhead = daysAhead,
            config = config
        )

        // Advanced analytics for debug logging
        if (enableDebugLogging) {
            logAdvancedAnalytics(simulationResults, currentPrice, volatility, daysAhead)
        }

        return """
            Upper band of ${String.format("%.2f", upperBand)} probability: ${String.format("%.1f", probUpper)}%
            Lower band of ${String.format("%.2f", lowerBand)} probability: ${String.format("%.1f", probLower)}%
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
        if (returns.size < 2) return 0.02

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.sum() / (returns.size - 1)
        return sqrt(variance)
    }

    private fun runAdvancedMonteCarloSimulation(
        currentPrice: Double,
        drift: Double,
        volatility: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        config: AdvancedMCConfig
    ): Triple<Double, Double, List<SimulationResult>> {

        val allResults = mutableListOf<SimulationResult>()
        val baseSimulations = if (config.useAntitheticVariates) config.numSimulations / 2 else config.numSimulations

        repeat(baseSimulations) { simIndex ->
            // Generate base simulation
            val baseResult = runSingleSimulation(
                currentPrice, drift, volatility, upperBand, lowerBand, daysAhead, simIndex, config
            )
            allResults.add(baseResult)

            // Generate antithetic variate if enabled
            if (config.useAntitheticVariates) {
                val antitheticResult = runAntitheticSimulation(
                    currentPrice, drift, volatility, upperBand, lowerBand, daysAhead, simIndex, config
                )
                allResults.add(antitheticResult)
            }
        }

        // Apply control variates if enabled
        val adjustedResults = if (config.useControlVariates) {
            applyControlVariates(allResults, currentPrice, drift, volatility, daysAhead)
        } else {
            allResults
        }

        // Calculate probabilities for touching bands at any point
        val reachesUpper = adjustedResults.count { it.hitsUpperBand }
        val reachesLower = adjustedResults.count { it.hitsLowerBand }

        val probUpper = (reachesUpper.toDouble() / adjustedResults.size) * 100
        val probLower = (reachesLower.toDouble() / adjustedResults.size) * 100

        return Triple(probUpper, probLower, adjustedResults)
    }

    private data class SimulationResult(
        val finalPrice: Double,
        val maxPrice: Double,
        val minPrice: Double,
        val hitsUpperBand: Boolean,
        val hitsLowerBand: Boolean,
        val pnl: Double,
        val path: List<Double>
    )

    private fun runSingleSimulation(
        currentPrice: Double,
        drift: Double,
        volatility: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        simIndex: Int,
        config: AdvancedMCConfig
    ): SimulationResult {

        val randomNumbers = if (config.useStratifiedSampling) {
            generateStratifiedRandomNumbers(daysAhead, simIndex, config.numStrata)
        } else {
            generateStandardRandomNumbers(daysAhead)
        }

        val path = geometricBrownianMotion(
            startPrice = currentPrice,
            drift = drift,
            volatility = volatility,
            timeSteps = daysAhead,
            randomNumbers = randomNumbers
        )

        val maxPrice = path.maxOrNull() ?: currentPrice
        val minPrice = path.minOrNull() ?: currentPrice
        val finalPrice = path.last()

        return SimulationResult(
            finalPrice = finalPrice,
            maxPrice = maxPrice,
            minPrice = minPrice,
            hitsUpperBand = path.any { it >= upperBand }, // Touch at any point
            hitsLowerBand = path.any { it <= lowerBand }, // Touch at any point
            pnl = finalPrice - currentPrice,
            path = path
        )
    }

    private fun runAntitheticSimulation(
        currentPrice: Double,
        drift: Double,
        volatility: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        baseSimIndex: Int,
        config: AdvancedMCConfig
    ): SimulationResult {

        // Use negative of the base random numbers for antithetic variates
        val baseRandomNumbers = if (config.useStratifiedSampling) {
            generateStratifiedRandomNumbers(daysAhead, baseSimIndex, config.numStrata)
        } else {
            generateStandardRandomNumbers(daysAhead)
        }

        val antitheticRandomNumbers = antitheticVariates(baseRandomNumbers)

        val path = geometricBrownianMotion(
            startPrice = currentPrice,
            drift = drift,
            volatility = volatility,
            timeSteps = daysAhead,
            randomNumbers = antitheticRandomNumbers
        )

        val maxPrice = path.maxOrNull() ?: currentPrice
        val minPrice = path.minOrNull() ?: currentPrice
        val finalPrice = path.last()

        return SimulationResult(
            finalPrice = finalPrice,
            maxPrice = maxPrice,
            minPrice = minPrice,
            hitsUpperBand = path.any { it >= upperBand }, // Touch at any point
            hitsLowerBand = path.any { it <= lowerBand }, // Touch at any point
            pnl = finalPrice - currentPrice,
            path = path
        )
    }

    private fun generateStandardRandomNumbers(count: Int): List<Double> {
        return (1..count).map { boxMuller() }
    }

    private fun generateStratifiedRandomNumbers(count: Int, simIndex: Int, numStrata: Int): List<Double> {
        val stratifiedNumbers = mutableListOf<Double>()

        repeat(count) { step ->
            val stratum = (simIndex + step) % numStrata
            val stratumWidth = 1.0 / numStrata
            val stratumStart = stratum * stratumWidth
            val uniform = stratumStart + Random.nextDouble() * stratumWidth

            // Convert uniform to normal using inverse normal CDF approximation
            val normal = inverseNormalCDF(uniform)
            stratifiedNumbers.add(normal)
        }

        return stratifiedNumbers
    }

    private fun antitheticVariates(normalRandoms: List<Double>): List<Double> {
        // Generate antithetic variates for variance reduction
        return normalRandoms.map { -it }
    }

    private fun applyControlVariates(
        results: List<SimulationResult>,
        currentPrice: Double,
        drift: Double,
        volatility: Double,
        daysAhead: Int
    ): List<SimulationResult> {

        // FIXED: Use daily scaling for control variate
        val analyticalExpectation = currentPrice * exp(drift * daysAhead)
        val empiricalMean = results.map { it.finalPrice }.average()

        // Calculate optimal control coefficient
        val covariance = results.map { (it.finalPrice - empiricalMean) * (it.finalPrice - analyticalExpectation) }.average()
        val controlVariance = results.map { (it.finalPrice - analyticalExpectation).pow(2) }.average()
        val optimalCoefficient = if (controlVariance > 0) -covariance / controlVariance else 0.0

        // Adjust results using control variate
        return results.map { result ->
            val adjustment = optimalCoefficient * (result.finalPrice - analyticalExpectation)
            val adjustedFinalPrice = result.finalPrice + adjustment

            result.copy(
                finalPrice = adjustedFinalPrice,
                pnl = adjustedFinalPrice - currentPrice
            )
        }
    }

    private fun geometricBrownianMotion(
        startPrice: Double,
        drift: Double,
        volatility: Double,
        timeSteps: Int,
        randomNumbers: List<Double>
    ): List<Double> {

        val dt = 1.0 // FIXED: Daily time step for daily data
        val prices = mutableListOf<Double>()
        var currentPrice = startPrice

        prices.add(currentPrice)

        for (i in 0 until minOf(timeSteps, randomNumbers.size)) {
            val normalRand = randomNumbers[i]
            // FIXED: Proper GBM formula with daily scaling
            val priceChange = exp((drift - 0.5 * volatility.pow(2)) * dt + volatility * sqrt(dt) * normalRand)
            currentPrice *= priceChange
            prices.add(currentPrice)
        }

        return prices
    }

    private fun boxMuller(): Double {
        var u = 0.0
        var v = 0.0

        while (u == 0.0) u = Random.nextDouble()
        while (v == 0.0) v = Random.nextDouble()

        return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
    }

    private fun inverseNormalCDF(probability: Double): Double {
        // Approximation of inverse normal CDF for stratified sampling
        if (probability <= 0.0) return Double.NEGATIVE_INFINITY
        if (probability >= 1.0) return Double.POSITIVE_INFINITY
        if (probability == 0.5) return 0.0

        val a = 2.515517
        val b = 0.802853
        val c = 0.010328
        val d1 = 1.432788
        val d2 = 0.189269
        val d3 = 0.001308

        val p = if (probability > 0.5) 1.0 - probability else probability
        val t = sqrt(-2.0 * ln(p))

        val numerator = a + b * t + c * t.pow(2)
        val denominator = 1.0 + d1 * t + d2 * t.pow(2) + d3 * t.pow(3)
        val result = t - numerator / denominator

        return if (probability > 0.5) result else -result
    }

    // Debug logging method
    private fun logDebugInfo(currentPrice: Double, meanReturn: Double, volatility: Double, drift: Double, daysAhead: Int) {
        Log.d("MonteCarloAdvanced", "=== Advanced Monte Carlo Parameters ===")
        Log.d("MonteCarloAdvanced", "Current Price: $currentPrice")
        Log.d("MonteCarloAdvanced", "Historical Mean Return (Daily): $meanReturn")
        Log.d("MonteCarloAdvanced", "Daily Volatility: $volatility")
        Log.d("MonteCarloAdvanced", "Annual Volatility: ${volatility * sqrt(252.0)}")
        Log.d("MonteCarloAdvanced", "Drift (Daily): $drift")
        Log.d("MonteCarloAdvanced", "Days Ahead: $daysAhead")
        Log.d("MonteCarloAdvanced", "Number of Simulations: ${AdvancedMCConfig().numSimulations}")
        Log.d("MonteCarloAdvanced", "Variance Reduction Techniques:")
        Log.d("MonteCarloAdvanced", "  - Antithetic Variates: ${AdvancedMCConfig().useAntitheticVariates}")
        Log.d("MonteCarloAdvanced", "  - Control Variates: ${AdvancedMCConfig().useControlVariates}")
        Log.d("MonteCarloAdvanced", "  - Stratified Sampling: ${AdvancedMCConfig().useStratifiedSampling}")
    }

    private fun logAdvancedAnalytics(results: List<SimulationResult>, currentPrice: Double, volatility: Double, daysAhead: Int) {
        val varResult = calculateVaR(results)
        val greeks = calculateGreeks(results, currentPrice, volatility, daysAhead)

        Log.d("MonteCarloAdvanced", "=== Advanced Analytics ===")
        Log.d("MonteCarloAdvanced", "Value at Risk:")
        Log.d("MonteCarloAdvanced", "  - 95% VaR: ${String.format("%.2f", varResult.var95)}")
        Log.d("MonteCarloAdvanced", "  - 99% VaR: ${String.format("%.2f", varResult.var99)}")
        Log.d("MonteCarloAdvanced", "  - Expected Shortfall: ${String.format("%.2f", varResult.expectedShortfall)}")
        Log.d("MonteCarloAdvanced", "  - Max Drawdown: ${String.format("%.2f%%", varResult.maxDrawdown * 100)}")

        Log.d("MonteCarloAdvanced", "Greeks:")
        Log.d("MonteCarloAdvanced", "  - Delta: ${String.format("%.4f", greeks.delta)}")
        Log.d("MonteCarloAdvanced", "  - Gamma: ${String.format("%.4f", greeks.gamma)}")
        Log.d("MonteCarloAdvanced", "  - Vega: ${String.format("%.4f", greeks.vega)}")
        Log.d("MonteCarloAdvanced", "  - Theta: ${String.format("%.4f", greeks.theta)}")

        val finalPrices = results.map { it.finalPrice }
        Log.d("MonteCarloAdvanced", "Simulation Statistics:")
        Log.d("MonteCarloAdvanced", "  - Mean Final Price: ${String.format("%.2f", finalPrices.average())}")
        Log.d("MonteCarloAdvanced", "  - Std Dev: ${String.format("%.2f", sqrt(finalPrices.map { (it - finalPrices.average()).pow(2) }.average()))}")
        Log.d("MonteCarloAdvanced", "  - Min Price: ${String.format("%.2f", finalPrices.minOrNull() ?: 0.0)}")
        Log.d("MonteCarloAdvanced", "  - Max Price: ${String.format("%.2f", finalPrices.maxOrNull() ?: 0.0)}")
    }

    private fun calculateGreeks(
        simulations: List<SimulationResult>,
        basePrice: Double,
        volatility: Double,
        daysAhead: Int
    ): Greeks {

        val timeToExpiry = daysAhead / 252.0
        val priceShift = basePrice * 0.01 // 1% price shift for finite difference
        val volShift = 0.01 // 1% volatility shift

        // Simplified Greeks calculation using finite differences
        val basePnL = simulations.map { it.pnl }.average()

        // Delta: sensitivity to price changes
        val delta = basePnL / priceShift

        // Gamma: second derivative with respect to price (simplified)
        val gamma = 0.0 // Would need additional simulations for accurate calculation

        // Vega: sensitivity to volatility changes
        val vega = basePnL * sqrt(timeToExpiry) * 0.1 // Simplified approximation

        // Theta: time decay (simplified)
        val theta = -basePnL / (timeToExpiry * 365) // Daily theta

        return Greeks(
            delta = delta,
            gamma = gamma,
            vega = vega,
            theta = theta
        )
    }

    private fun calculateVaR(results: List<SimulationResult>): VaRResult {
        val pnls = results.map { it.pnl }.sorted()
        val n = pnls.size

        val var95 = pnls[(n * 0.05).toInt()] // 5th percentile (95% VaR)
        val var99 = pnls[(n * 0.01).toInt()] // 1st percentile (99% VaR)

        // Expected Shortfall (average of worst 5% outcomes)
        val worstOutcomes = pnls.take((n * 0.05).toInt())
        val expectedShortfall = if (worstOutcomes.isNotEmpty()) worstOutcomes.average() else 0.0

        // Maximum drawdown from paths
        val maxDrawdown = results.map { result ->
            var maxDrawdownForPath = 0.0
            var peak = result.path.first()

            for (price in result.path) {
                if (price > peak) peak = price
                val drawdown = (peak - price) / peak
                if (drawdown > maxDrawdownForPath) maxDrawdownForPath = drawdown
            }
            maxDrawdownForPath
        }.maxOrNull() ?: 0.0

        return VaRResult(
            var95 = var95,
            var99 = var99,
            expectedShortfall = expectedShortfall,
            maxDrawdown = maxDrawdown
        )
    }
}