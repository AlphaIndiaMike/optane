package com.alphaindiamike.optane.algorithms.implementations


import kotlin.math.*
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations
import android.util.Log

/**
 * GARCHForecaster - Fixed Implementation
 * GARCH-based volatility forecasting with proper parameter estimation and barrier probability calculation
 */
class GARCHForecasterImpl(private val enableDebugLogging: Boolean = true) : AlgorithmRepository {

    private data class GARCHParams(
        val omega: Double,    // Long-term variance
        val alpha: Double,    // ARCH effect
        val beta: Double,     // GARCH effect
        val mean: Double
    )

    private data class VolatilityForecast(
        val day: Int,
        val variance: Double,
        val volatility: Double,
        val annualizedVol: Double
    )

    private lateinit var returns: List<Double>
    private lateinit var garchParams: GARCHParams

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
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
            // 1. Calculate log returns
            returns = calculateReturns(timeSeries)

            // 2. Estimate GARCH(1,1) parameters using MLE
            garchParams = estimateGARCHByMLE(returns)

            // 3. Forecast volatility for specified days
            val volForecasts = forecastVolatility(daysAhead, garchParams)

            // 4. Calculate first-passage probabilities
            val currentPrice = timeSeries.last().price
            val upperProbability = calculateFirstPassageProbability(
                currentPrice = currentPrice,
                barrier = upperBand,
                timeHorizon = daysAhead.toDouble(),
                volForecasts = volForecasts,
                isUpper = true
            )

            val lowerProbability = calculateFirstPassageProbability(
                currentPrice = currentPrice,
                barrier = lowerBand,
                timeHorizon = daysAhead.toDouble(),
                volForecasts = volForecasts,
                isUpper = false
            )

            if (enableDebugLogging) {
                logDebugInfo(currentPrice, upperBand, lowerBand, daysAhead, volForecasts, upperProbability, lowerProbability)
            }

            return """
                Upper band of ${upperBand} probability: ${String.format("%.1f", upperProbability * 100)}%
                Lower band of ${lowerBand} probability: ${String.format("%.1f", lowerProbability * 100)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("GARCH", "Error in calculation: ${e.message}")
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

    /**
     * Improved GARCH parameter estimation using Maximum Likelihood Estimation
     */
    private fun estimateGARCHByMLE(returns: List<Double>): GARCHParams {
        if (returns.size < 10) {
            // Fallback to simple estimation for insufficient data
            return estimateGARCHSimple(returns)
        }

        val mean = returns.average()
        val residuals = returns.map { it - mean }

        var bestLogLikelihood = Double.NEGATIVE_INFINITY
        var bestParams = GARCHParams(0.01, 0.1, 0.8, mean)

        // Grid search for optimal parameters
        val omegaValues = listOf(0.0001, 0.001, 0.01, 0.05)
        val alphaValues = listOf(0.05, 0.1, 0.15, 0.2, 0.25)
        val betaValues = listOf(0.6, 0.7, 0.8, 0.85, 0.9)

        for (omega in omegaValues) {
            for (alpha in alphaValues) {
                for (beta in betaValues) {
                    // Stationarity condition: alpha + beta < 1
                    if (alpha + beta < 0.98 && alpha > 0 && beta > 0 && omega > 0) {
                        val params = GARCHParams(omega, alpha, beta, mean)
                        val logLikelihood = calculateLogLikelihood(residuals, params)

                        if (logLikelihood > bestLogLikelihood) {
                            bestLogLikelihood = logLikelihood
                            bestParams = params
                        }
                    }
                }
            }
        }

        return bestParams
    }

    private fun estimateGARCHSimple(returns: List<Double>): GARCHParams {
        val mean = returns.average()
        val residuals = returns.map { it - mean }
        val initialVariance = residuals.map { it.pow(2) }.average()

        return GARCHParams(
            omega = initialVariance * 0.1,
            alpha = 0.1,
            beta = 0.8,
            mean = mean
        )
    }

    private fun calculateLogLikelihood(residuals: List<Double>, params: GARCHParams): Double {
        if (residuals.size < 2) return Double.NEGATIVE_INFINITY

        var logLikelihood = 0.0

        // Initialize variance with unconditional variance
        var variance = params.omega / (1 - params.alpha - params.beta)
        if (variance <= 0) return Double.NEGATIVE_INFINITY

        for (i in 1 until residuals.size) {
            // GARCH(1,1) variance evolution
            variance = params.omega + params.alpha * residuals[i-1].pow(2) + params.beta * variance

            if (variance <= 0) return Double.NEGATIVE_INFINITY

            // Log-likelihood contribution
            logLikelihood += -0.5 * ln(2 * PI) - 0.5 * ln(variance) - 0.5 * residuals[i].pow(2) / variance
        }

        return logLikelihood
    }

    /**
     * Fixed volatility forecasting with proper variance evolution
     */
    private fun forecastVolatility(daysAhead: Int, garchParams: GARCHParams): List<VolatilityForecast> {
        val forecasts = mutableListOf<VolatilityForecast>()

        // Initialize with the most recent conditional variance
        var currentVariance = if (returns.size >= 2) {
            // Use last observed squared residual and previous variance estimate
            val lastResidual = returns.last() - garchParams.mean
            val penultimateResidual = returns[returns.size - 2] - garchParams.mean

            // Estimate previous variance using GARCH equation solved backwards
            val unconditionalVar = garchParams.omega / (1 - garchParams.alpha - garchParams.beta)
            garchParams.omega + garchParams.alpha * penultimateResidual.pow(2) + garchParams.beta * unconditionalVar
        } else {
            garchParams.omega / (1 - garchParams.alpha - garchParams.beta)
        }

        // Get the last residual for the first forecast step
        val lastResidual = if (returns.isNotEmpty()) {
            returns.last() - garchParams.mean
        } else {
            0.0
        }

        for (day in 1..daysAhead) {
            val nextVariance = if (day == 1) {
                // First day: use actual last residual
                garchParams.omega + garchParams.alpha * lastResidual.pow(2) + garchParams.beta * currentVariance
            } else {
                // Subsequent days: use unconditional expectation for future residuals
                val unconditionalVar = garchParams.omega / (1 - garchParams.alpha - garchParams.beta)
                garchParams.omega + garchParams.alpha * unconditionalVar + garchParams.beta * currentVariance
            }

            val nextVolatility = sqrt(maxOf(nextVariance, 1e-8)) // Prevent negative variance
            val annualizedVol = nextVolatility * sqrt(252.0)

            forecasts.add(VolatilityForecast(
                day = day,
                variance = nextVariance,
                volatility = nextVolatility,
                annualizedVol = annualizedVol
            ))

            currentVariance = nextVariance
        }

        return forecasts
    }

    /**
     * Calculate first-passage probability using GARCH-forecasted volatility
     * This calculates P(touch barrier at any point during timeHorizon)
     */
    /*
    private fun calculateFirstPassageProbability(
        currentPrice: Double,
        barrier: Double,
        timeHorizon: Double,
        volForecasts: List<VolatilityForecast>,
        isUpper: Boolean
    ): Double {

        // Check if already past barrier
        if ((isUpper && currentPrice >= barrier) || (!isUpper && currentPrice <= barrier)) {
            return 1.0
        }

        if (timeHorizon <= 0.0) return 0.0

        // Calculate cumulative volatility over the forecast period
        //val cumulativeVariance = volForecasts.sumOf { it.variance }
        //val totalVolatility = sqrt(cumulativeVariance)
        val avgDailyVol = volForecasts.map { it.volatility }.average()
        val totalVolatility = avgDailyVol * sqrt(timeHorizon)

        if (totalVolatility <= 0.0) return 0.0

        // Use the mean drift from GARCH estimation
        val drift = garchParams.mean

        // Calculate first-passage probability using analytical formula
        return calculateFirstPassageAnalytical(
            startPrice = currentPrice,
            barrier = barrier,
            drift = drift,
            volatility = totalVolatility,
            timeHorizon = timeHorizon,
            isUpper = isUpper
        )
    }*/

    private fun calculateFirstPassageProbability(
        currentPrice: Double,
        barrier: Double,
        timeHorizon: Double,
        volForecasts: List<VolatilityForecast>,
        isUpper: Boolean
    ): Double {

        // Check if already past barrier
        if ((isUpper && currentPrice >= barrier) || (!isUpper && currentPrice <= barrier)) {
            return 1.0
        }

        val numSimulations = 5000
        var touchCount = 0
        val numSteps = timeHorizon.toInt()

        repeat(numSimulations) {
            var currentPrice = currentPrice
            var touched = false

            for (step in 1..numSteps) {
                // Get volatility for this step (or use last available)
                val dayVol = volForecasts.getOrNull(step - 1)?.volatility
                    ?: volForecasts.lastOrNull()?.volatility
                    ?: 0.02

                // Generate random normal
                val random = generateNormalRandom()

                // Update price using GBM with GARCH volatility
                val priceChange = garchParams.mean + dayVol * random
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

    private fun generateNormalRandom(): Double {
        val u1 = kotlin.random.Random.Default.nextDouble()
        val u2 = kotlin.random.Random.Default.nextDouble()
        return sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    /**
     * Analytical first-passage probability for Geometric Brownian Motion
     */
    private fun calculateFirstPassageAnalytical(
        startPrice: Double,
        barrier: Double,
        drift: Double,
        volatility: Double,
        timeHorizon: Double,
        isUpper: Boolean
    ): Double {

        if (timeHorizon <= 0.0 || volatility <= 0.0) return 0.0

        val logRatio = ln(barrier / startPrice)
        val mu = drift - 0.5 * volatility * volatility
        val sigmaSqrtT = volatility * sqrt(timeHorizon)

        if (abs(logRatio) < 1e-10) return 1.0 // Already at barrier

        // First-passage probability formula
        val d1 = (-logRatio + mu * timeHorizon) / sigmaSqrtT
        val d2 = (-logRatio - mu * timeHorizon) / sigmaSqrtT

        val prob1 = if (isUpper) {
            1.0 - normalCDF(d1)
        } else {
            normalCDF(d1)
        }

        val alpha = 2.0 * mu / (volatility * volatility)
        val reflectionTerm = if (abs(alpha) < 20.0) { // Prevent overflow
            (barrier / startPrice).pow(alpha)
        } else {
            0.0
        }

        val prob2 = if (isUpper) {
            reflectionTerm * normalCDF(d2)
        } else {
            reflectionTerm * (1.0 - normalCDF(d2))
        }

        return minOf(1.0, maxOf(0.0, prob1 + prob2))
    }

    private fun normalCDF(x: Double): Double {
        if (x < -6.0) return 0.0
        if (x > 6.0) return 1.0

        // Abramowitz and Stegun approximation
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

    /**
     * Calculate volatility clustering metrics for validation
     */
    private fun calculateVolatilityClustering(): Map<String, Double> {
        if (returns.size < 10) return emptyMap()

        val residuals = returns.map { it - garchParams.mean }
        val squaredReturns = residuals.map { it.pow(2) }

        // Calculate autocorrelation of squared returns
        val autocorr1 = calculateAutocorrelation(squaredReturns, 1)
        val autocorr5 = calculateAutocorrelation(squaredReturns, 5)

        // Calculate persistence
        val persistence = garchParams.alpha + garchParams.beta

        // Calculate unconditional variance
        val unconditionalVar = garchParams.omega / (1 - persistence)

        return mapOf(
            "autocorr1" to autocorr1,
            "autocorr5" to autocorr5,
            "persistence" to persistence,
            "unconditionalVolatility" to sqrt(unconditionalVar * 252),
            "omega" to garchParams.omega,
            "alpha" to garchParams.alpha,
            "beta" to garchParams.beta
        )
    }

    private fun calculateAutocorrelation(series: List<Double>, lag: Int): Double {
        if (series.size <= lag) return 0.0

        val mean = series.average()
        val variance = series.map { (it - mean).pow(2) }.average()

        if (variance == 0.0) return 0.0

        var covariance = 0.0
        val n = series.size - lag

        for (i in 0 until n) {
            covariance += (series[i] - mean) * (series[i + lag] - mean)
        }

        covariance /= n
        return covariance / variance
    }

    private fun logDebugInfo(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        volForecasts: List<VolatilityForecast>,
        upperProb: Double,
        lowerProb: Double
    ) {
        Log.d("GARCH", "=== GARCH FORECASTER DEBUG INFO ===")
        Log.d("GARCH", "Current price: $currentPrice")
        Log.d("GARCH", "Upper band: $upperBand, Lower band: $lowerBand")
        Log.d("GARCH", "Days ahead: $daysAhead")
        Log.d("GARCH", "GARCH Parameters:")
        Log.d("GARCH", "  Omega: ${String.format("%.6f", garchParams.omega)}")
        Log.d("GARCH", "  Alpha: ${String.format("%.4f", garchParams.alpha)}")
        Log.d("GARCH", "  Beta: ${String.format("%.4f", garchParams.beta)}")
        Log.d("GARCH", "  Mean: ${String.format("%.6f", garchParams.mean)}")
        Log.d("GARCH", "  Persistence: ${String.format("%.4f", garchParams.alpha + garchParams.beta)}")

        if (volForecasts.isNotEmpty()) {
            Log.d("GARCH", "Volatility Forecasts:")
            volForecasts.take(5).forEach { forecast ->
                Log.d("GARCH", "  Day ${forecast.day}: ${String.format("%.4f", forecast.volatility)} " +
                        "(${String.format("%.1f", forecast.annualizedVol * 100)}% annual)")
            }
        }

        Log.d("GARCH", "Results:")
        Log.d("GARCH", "  Upper probability: ${String.format("%.1f", upperProb * 100)}%")
        Log.d("GARCH", "  Lower probability: ${String.format("%.1f", lowerProb * 100)}%")
        Log.d("GARCH", "=== END DEBUG INFO ===")
    }
}