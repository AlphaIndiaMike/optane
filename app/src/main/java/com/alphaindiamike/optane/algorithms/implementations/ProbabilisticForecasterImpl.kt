package com.alphaindiamike.optane.algorithms.implementations

import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import kotlin.math.*
import com.alphaindiamike.optane.model.Calculations

/**
 * Default Model: Finance School - ProbabilisticForecaster
 * Basic probabilistic forecasting using log-normal distribution
 */
class ProbabilisticForecasterImpl : AlgorithmRepository {

    // Algorithm-specific configuration (internal)
    private data class Config(
        val probabilityLevels: List<Double> = listOf(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    )

    // Core calculation data
    private var dailyReturns: List<Double> = emptyList()
    private var avgReturn: Double = 0.0
    private var volatility: Double = 0.0
    private var currentPrice: Double = 0.0

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 2) {
            return "Insufficient data"
        }

        // Initialize with price data
        currentPrice = timeSeries.last().price
        dailyReturns = calculateReturns(timeSeries)
        avgReturn = calculateMean(dailyReturns)
        volatility = calculateVolatility()

        // Calculate probabilities for target bands
        val upperProbability = calculateBandProbability(upperBand, daysAhead, true)
        val lowerProbability = calculateBandProbability(lowerBand, daysAhead, false)

        return """
            Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", upperProbability)}%
            Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", lowerProbability)}%
            """.trimIndent()
    }

    private fun calculateReturns(priceData: List<TimeSeriesEntity>): List<Double> {
        val returns = mutableListOf<Double>()
        for (i in 1 until priceData.size) {
            val return_ = (priceData[i].price - priceData[i-1].price) / priceData[i-1].price
            returns.add(return_)
        }
        return returns
    }

    private fun calculateMean(arr: List<Double>): Double {
        return if (arr.isNotEmpty()) arr.sum() / arr.size else 0.0
    }

    private fun calculateVolatility(): Double {
        if (dailyReturns.size <= 1) return 0.02 // Default 2% volatility

        val variance = dailyReturns.map { r ->
            (r - avgReturn).pow(2)
        }.sum() / (dailyReturns.size - 1)

        return sqrt(variance)
    }

    private fun calculateBandProbability(targetPrice: Double, daysAhead: Int, isUpperBand: Boolean): Double {
        // Calculate the probability that price will reach the target within daysAhead days
        val expectedLogReturn = avgReturn * daysAhead
        val totalVolatility = volatility * sqrt(daysAhead.toDouble())

        // Calculate the required return to reach target
        val requiredLogReturn = ln(targetPrice / currentPrice)

        // Calculate z-score
        val z = (requiredLogReturn - expectedLogReturn) / totalVolatility

        // Calculate probability using cumulative normal distribution
        val probability = if (isUpperBand) {
            // For upper band: P(price >= target) = 1 - Φ(z)
            1.0 - normalCDF(z)
        } else {
            // For lower band: P(price <= target) = Φ(z)
            normalCDF(z)
        }

        return probability * 100.0 // Convert to percentage
    }

    private fun normalCDF(x: Double): Double {
        // Standard normal cumulative distribution function
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

    // Inverse normal CDF approximation using Beasley-Springer-Moro algorithm
    private fun getZScore(probability: Double): Double {
        if (probability == 0.5) return 0.0

        // Beasley-Springer-Moro algorithm coefficients
        val a = doubleArrayOf(0.0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        val b = doubleArrayOf(0.0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01)
        val c = doubleArrayOf(0.0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
        val d = doubleArrayOf(0.0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00)

        val pLow = 0.02425
        val pHigh = 1 - pLow

        return when {
            probability < pLow -> {
                val q = sqrt(-2 * ln(probability))
                (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                        ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
            }
            probability <= pHigh -> {
                val q = probability - 0.5
                val r = q * q
                (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
                        (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
            }
            else -> {
                val q = sqrt(-2 * ln(1 - probability))
                -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                        ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
            }
        }
    }

    // Main forecasting method (following your JS structure)
    private fun forecast(daysAhead: Int, probabilityLevels: List<Double> = Config().probabilityLevels): Map<String, Map<String, Double>> {
        val results = mutableMapOf<String, Map<String, Double>>()

        for (day in 1..daysAhead) {
            val expectedLogReturn = avgReturn * day
            val totalVolatility = volatility * sqrt(day.toDouble())

            val dayResults = mutableMapOf<String, Double>()

            probabilityLevels.forEach { prob ->
                val z = getZScore(prob)
                val logPrice = ln(currentPrice) + expectedLogReturn + z * totalVolatility
                val price = exp(logPrice)
                dayResults["${(prob * 100).roundToInt()}%"] = (price * 100).roundToInt() / 100.0
            }

            results["day_$day"] = dayResults
        }

        return results
    }

    // Convenient method for specific probability format (from your JS)
    private fun getOddsFormat(day: Int, targetProbabilities: List<Double> = listOf(0.9, 0.6, 0.3, 0.01)): Map<String, Double> {
        val result = mutableMapOf<String, Double>()

        targetProbabilities.forEach { prob ->
            val z = getZScore(prob)
            val expectedLogReturn = avgReturn * day
            val totalVolatility = volatility * sqrt(day.toDouble())
            val logPrice = ln(currentPrice) + expectedLogReturn + z * totalVolatility
            val price = exp(logPrice)
            result["${(prob * 100).roundToInt()}%"] = (price * 100).roundToInt() / 100.0
        }

        return result
    }

    // Calculate specific probability for a target price (utility method)
    private fun calculateTargetProbability(targetPrice: Double, daysAhead: Int): Double {
        val expectedLogReturn = avgReturn * daysAhead
        val totalVolatility = volatility * sqrt(daysAhead.toDouble())
        val logPrice = ln(targetPrice / currentPrice)

        // Calculate z-score
        val z = (logPrice - expectedLogReturn) / totalVolatility

        // Return cumulative probability
        return normalCDF(z)
    }

    // Additional utility methods for debugging/analysis

    private fun getModelParameters(): Map<String, Double> {
        return mapOf(
            "currentPrice" to currentPrice,
            "avgDailyReturn" to avgReturn,
            "dailyVolatility" to volatility,
            "annualVolatility" to (volatility * sqrt(252.0)),
            "numberOfObservations" to dailyReturns.size.toDouble()
        )
    }

    private fun getConfidenceInterval(daysAhead: Int, confidenceLevel: Double = 0.95): Pair<Double, Double> {
        val alpha = 1.0 - confidenceLevel
        val lowerProb = alpha / 2.0
        val upperProb = 1.0 - alpha / 2.0

        val expectedLogReturn = avgReturn * daysAhead
        val totalVolatility = volatility * sqrt(daysAhead.toDouble())

        val lowerZ = getZScore(lowerProb)
        val upperZ = getZScore(upperProb)

        val lowerPrice = currentPrice * exp(expectedLogReturn + lowerZ * totalVolatility)
        val upperPrice = currentPrice * exp(expectedLogReturn + upperZ * totalVolatility)

        return Pair(lowerPrice, upperPrice)
    }
}