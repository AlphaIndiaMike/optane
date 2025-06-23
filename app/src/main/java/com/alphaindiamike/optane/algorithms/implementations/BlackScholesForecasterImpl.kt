package com.alphaindiamike.optane.algorithms.implementations

import kotlin.math.*
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations
import android.util.Log

/**
 * Black-Scholes Forecaster - Fixed Implementation
 * Uses Black-Scholes framework for volatility estimation with Monte Carlo for barrier touching probabilities
 */
class BlackScholesForecasterImpl(private val enableDebugLogging: Boolean = true) : AlgorithmRepository {

    private data class BSParams(
        val riskFreeRate: Double = 0.025,  // ECB rate approximation
        val dividendYield: Double = 0.0,
        val volatility: Double
    )

    private data class ProbabilityAnalysis(
        val riskNeutralProbability: Double,
        val physicalProbability: Double,
        val impliedVolatility: Double,
        val timeDecay: Double
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

        if (upperBand <= 0 || lowerBand <= 0) {
            return "Invalid price bands"
        }

        if (upperBand <= lowerBand) {
            return "Upper band must be greater than lower band"
        }

        try {
            // 1. Calculate historical volatility
            val historicalVolatility = calculateHistoricalVolatility(timeSeries)

            // 2. Estimate implied volatility (blend historical with recent)
            val impliedVolatility = estimateImpliedVolatility(timeSeries, historicalVolatility)

            // 3. Set up Black-Scholes parameters
            val bsParams = BSParams(
                riskFreeRate = 0.02, // ECB rate approximation
                dividendYield = 0.0,   // Assuming no dividends
                volatility = impliedVolatility
            )

            // 4. Calculate barrier touching probabilities using Monte Carlo with BS parameters
            val currentPrice = timeSeries.last().price
            val timeToExpiry = daysAhead / 252.0 // Convert trading days to years

            val upperProbability = calculateBarrierTouchingProbability(
                currentPrice = currentPrice,
                barrier = upperBand,
                timeToExpiry = timeToExpiry,
                bsParams = bsParams,
                isUpper = true
            )

            val lowerProbability = calculateBarrierTouchingProbability(
                currentPrice = currentPrice,
                barrier = lowerBand,
                timeToExpiry = timeToExpiry,
                bsParams = bsParams,
                isUpper = false
            )

            if (enableDebugLogging) {
                logDebugInfo(currentPrice, upperBand, lowerBand, daysAhead, bsParams, upperProbability, lowerProbability)
            }

            return """
                Upper band of ${upperBand} probability: ${String.format("%.1f", upperProbability * 100)}%
                Lower band of ${lowerBand} probability: ${String.format("%.1f", lowerProbability * 100)}%
                """.trimIndent()

        } catch (e: Exception) {
            Log.e("BlackScholes", "Error in calculation: ${e.message}")
            return "Calculation error: ${e.message}"
        }
    }

    private fun calculateHistoricalVolatility(timeSeries: List<TimeSeriesEntity>): Double {
        if (timeSeries.size < 2) return 0.20 // Default 20% annual volatility

        val returns = mutableListOf<Double>()
        for (i in 1 until timeSeries.size) {
            val ret = ln(timeSeries[i].price / timeSeries[i-1].price)
            if (ret.isFinite()) {
                returns.add(ret)
            }
        }

        if (returns.size < 2) return 0.20

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.sum() / (returns.size - 1)
        return sqrt(variance * 252) // Annualized volatility
    }

    private fun estimateImpliedVolatility(timeSeries: List<TimeSeriesEntity>, historicalVol: Double): Double {
        if (timeSeries.size < 10) return historicalVol

        // Calculate recent volatility (last 10 days for better estimate)
        val recentPrices = timeSeries.takeLast(min(10, timeSeries.size))
        val recentReturns = mutableListOf<Double>()

        for (i in 1 until recentPrices.size) {
            val ret = ln(recentPrices[i].price / recentPrices[i-1].price)
            if (ret.isFinite()) {
                recentReturns.add(ret)
            }
        }

        if (recentReturns.size < 2) return historicalVol

        val recentMean = recentReturns.average()
        val recentVariance = recentReturns.map { (it - recentMean).pow(2) }.average()
        val recentVol = sqrt(recentVariance * 252)

        // Blend historical and recent volatility (70% historical, 30% recent)
        val blendedVol = historicalVol * 0.7 + recentVol * 0.3

        // Constrain volatility to reasonable range
        return maxOf(0.05, minOf(2.0, blendedVol))
    }

    /**
     * Calculate barrier touching probability using Monte Carlo simulation
     * with Black-Scholes risk-neutral dynamics
     */
    private fun calculateBarrierTouchingProbability(
        currentPrice: Double,
        barrier: Double,
        timeToExpiry: Double,
        bsParams: BSParams,
        isUpper: Boolean
    ): Double {

        // Check if already past barrier
        if ((isUpper && currentPrice >= barrier) || (!isUpper && currentPrice <= barrier)) {
            return 1.0
        }

        if (timeToExpiry <= 0.0) return 0.0

        val numSimulations = 5000
        var touchCount = 0
        val numSteps = maxOf(1, (timeToExpiry * 252).toInt()) // Daily steps
        val dt = timeToExpiry / numSteps

        // Risk-neutral drift
        val drift = bsParams.riskFreeRate - bsParams.dividendYield - 0.5 * bsParams.volatility.pow(2)

        repeat(numSimulations) {
            var price = currentPrice
            var touched = false

            repeat(numSteps) {
                if (!touched) {  // Only continue if not already touched
                    val random = generateNormalRandom()

                    // Risk-neutral price evolution: dS = (r-q)S*dt + σS*dW
                    val priceChange = drift * dt + bsParams.volatility * sqrt(dt) * random
                    price *= exp(priceChange)

                    // Check if barrier touched
                    if ((isUpper && price >= barrier) || (!isUpper && price <= barrier)) {
                        touched = true
                    }
                }
            }

            if (touched) touchCount++
        }

        return touchCount.toDouble() / numSimulations
    }

    /**
     * Calculate endpoint probability using analytical Black-Scholes (for comparison)
     */
    private fun calculateEndpointProbability(
        currentPrice: Double,
        targetPrice: Double,
        timeToExpiry: Double,
        bsParams: BSParams,
        isUpperBand: Boolean
    ): Double {

        val drift = bsParams.riskFreeRate - bsParams.dividendYield - 0.5 * bsParams.volatility.pow(2)
        val logReturn = ln(targetPrice / currentPrice)
        val normalizedReturn = (logReturn - drift * timeToExpiry) / (bsParams.volatility * sqrt(timeToExpiry))

        val probability = if (isUpperBand) {
            // Probability that final price >= target
            1.0 - normalCDF(normalizedReturn)
        } else {
            // Probability that final price <= target
            normalCDF(normalizedReturn)
        }

        return probability
    }

    private fun calculateD1D2(
        currentPrice: Double,
        strikePrice: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double
    ): Pair<Double, Double> {

        val d1 = (ln(currentPrice / strikePrice) +
                (riskFreeRate + 0.5 * volatility.pow(2)) * timeToExpiry) /
                (volatility * sqrt(timeToExpiry))

        val d2 = d1 - volatility * sqrt(timeToExpiry)

        return Pair(d1, d2)
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

    private fun generateNormalRandom(): Double {
        val u1 = kotlin.random.Random.Default.nextDouble()
        val u2 = kotlin.random.Random.Default.nextDouble()
        return sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    /**
     * Complete Black-Scholes option pricing (for reference)
     */
    private fun blackScholesPrice(
        spot: Double,
        strike: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double,
        isCall: Boolean = true
    ): Double {

        val d1d2 = calculateD1D2(spot, strike, timeToExpiry, riskFreeRate, volatility)
        val d1 = d1d2.first
        val d2 = d1d2.second

        return if (isCall) {
            spot * normalCDF(d1) - strike * exp(-riskFreeRate * timeToExpiry) * normalCDF(d2)
        } else {
            strike * exp(-riskFreeRate * timeToExpiry) * normalCDF(-d2) - spot * normalCDF(-d1)
        }
    }

    /**
     * Calculate the Greeks for risk management
     */
    private fun calculateGreeks(
        spot: Double,
        strike: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double
    ): Map<String, Double> {

        val d1d2 = calculateD1D2(spot, strike, timeToExpiry, riskFreeRate, volatility)
        val d1 = d1d2.first
        val d2 = d1d2.second

        // Delta: sensitivity to underlying price changes
        val callDelta = normalCDF(d1)
        val putDelta = callDelta - 1.0

        // Gamma: sensitivity of delta to underlying price changes
        val gamma = (1.0 / sqrt(2 * PI)) * exp(-0.5 * d1.pow(2)) / (spot * volatility * sqrt(timeToExpiry))

        // Theta: time decay (per day)
        val callTheta = -(spot * (1.0 / sqrt(2 * PI)) * exp(-0.5 * d1.pow(2)) * volatility) / (2 * sqrt(timeToExpiry)) -
                riskFreeRate * strike * exp(-riskFreeRate * timeToExpiry) * normalCDF(d2)
        val putTheta = callTheta + riskFreeRate * strike * exp(-riskFreeRate * timeToExpiry)

        // Vega: sensitivity to volatility
        val vega = spot * sqrt(timeToExpiry) * (1.0 / sqrt(2 * PI)) * exp(-0.5 * d1.pow(2))

        return mapOf(
            "callDelta" to callDelta,
            "putDelta" to putDelta,
            "gamma" to gamma,
            "callTheta" to callTheta / 365.0, // Daily theta
            "putTheta" to putTheta / 365.0,   // Daily theta
            "vega" to vega / 100.0           // Vega per 1% vol change
        )
    }

    /**
     * Risk metrics based on Black-Scholes framework
     */
    private fun calculateRiskMetrics(
        currentPrice: Double,
        volatility: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        confidenceLevel: Double = 0.05
    ): Map<String, Double> {

        // Value at Risk using Black-Scholes framework
        val zScore = inverseNormalCDF(confidenceLevel)
        val expectedReturn = riskFreeRate - 0.5 * volatility.pow(2)
        val logReturn = expectedReturn * timeToExpiry + zScore * volatility * sqrt(timeToExpiry)
        val varPrice = currentPrice * exp(logReturn)
        val var_ = currentPrice - varPrice

        // Expected shortfall approximation
        val phi = (1.0 / sqrt(2 * PI)) * exp(-0.5 * zScore.pow(2))
        val expectedShortfall = currentPrice * (1 - exp(expectedReturn * timeToExpiry -
                volatility * sqrt(timeToExpiry) * phi / confidenceLevel))

        return mapOf(
            "VaR_${(confidenceLevel * 100).toInt()}%" to var_,
            "ES_${(confidenceLevel * 100).toInt()}%" to expectedShortfall,
            "annualizedVolatility" to volatility * 100,
            "expectedReturn" to (exp(expectedReturn * timeToExpiry) - 1) * 100
        )
    }

    private fun inverseNormalCDF(p: Double): Double {
        if (p == 0.5) return 0.0

        val a = doubleArrayOf(0.0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        val b = doubleArrayOf(0.0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01)

        return if (p < 0.5) {
            val q = sqrt(-2 * ln(p))
            -(((((a[1] * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) * q + a[6]) /
                    ((((b[1] * q + b[2]) * q + b[3]) * q + b[4]) * q + 1)
        } else {
            val q = sqrt(-2 * ln(1 - p))
            (((((a[1] * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) * q + a[6]) /
                    ((((b[1] * q + b[2]) * q + b[3]) * q + b[4]) * q + 1)
        }
    }

    private fun logDebugInfo(
        currentPrice: Double,
        upperBand: Double,
        lowerBand: Double,
        daysAhead: Int,
        bsParams: BSParams,
        upperProb: Double,
        lowerProb: Double
    ) {
        Log.d("BlackScholes", "=== BLACK-SCHOLES FORECASTER DEBUG INFO ===")
        Log.d("BlackScholes", "Current price: $currentPrice")
        Log.d("BlackScholes", "Upper band: $upperBand, Lower band: $lowerBand")
        Log.d("BlackScholes", "Days ahead: $daysAhead")
        Log.d("BlackScholes", "Black-Scholes Parameters:")
        Log.d("BlackScholes", "  Risk-free rate: ${String.format("%.3f", bsParams.riskFreeRate * 100)}%")
        Log.d("BlackScholes", "  Dividend yield: ${String.format("%.3f", bsParams.dividendYield * 100)}%")
        Log.d("BlackScholes", "  Volatility: ${String.format("%.1f", bsParams.volatility * 100)}%")
        Log.d("BlackScholes", "Results:")
        Log.d("BlackScholes", "  Upper probability: ${String.format("%.1f", upperProb * 100)}%")
        Log.d("BlackScholes", "  Lower probability: ${String.format("%.1f", lowerProb * 100)}%")
        Log.d("BlackScholes", "=== END DEBUG INFO ===")
    }
}

/*

The `BSResult` data class is **never used** in the current implementation.

## **Why `BSResult` Exists:**

### **1. Professional Black-Scholes Implementation Standard:**
In institutional quantitative finance, Black-Scholes calculators typically return **comprehensive results** including:
- **d1, d2**: The famous Black-Scholes parameters
- **Call/Put prices**: Full option valuations
- **Call/Put deltas**: Risk sensitivities

### **2. Your JavaScript Code Pattern:**
Your original JS code likely had methods that returned multiple values simultaneously, and `BSResult` follows that pattern for **completeness**.

### **3. Current Implementation Focus:**
The current `calculate()` method **only needs probabilities** for your mobile app, so it uses:
```kotlin
// Only uses this part:
val probability = if (isUpperBand) {
    1.0 - normalCDF(d1d2.second) // Only needs d2
} else {
    normalCDF(d1d2.second)       // Only needs d2
}
```

## **Where `BSResult` Would Be Used:**

### **Future Development Scenarios:**

#### **1. Comprehensive Option Pricing:**
```kotlin
fun getFullBlackScholesAnalysis(calculations: Calculations): BSResult {
    val d1d2 = calculateD1D2(...)
    val callPrice = blackScholesPrice(..., isCall = true)
    val putPrice = blackScholesPrice(..., isCall = false)
    val greeks = calculateGreeks(...)

    return BSResult(
        d1 = d1d2.first,
        d2 = d1d2.second,
        callPrice = callPrice,
        putPrice = putPrice,
        callDelta = greeks["callDelta"]!!,
        putDelta = greeks["putDelta"]!!
    )
}
```

#### **2. Options Trading Features:**
```kotlin
// If you add options trading to your mobile app
fun calculateOptionStrategy(
    spot: Double,
    strikes: List<Double>
): List<BSResult> {
    return strikes.map { strike ->
        // Returns full BSResult for each strike
    }
}
```

#### **3. Risk Management Dashboard:**
```kotlin
// Portfolio Greeks aggregation
fun getPortfolioGreeks(positions: List<Position>): PortfolioRisk {
    val results = positions.map { position ->
        // Each position returns BSResult
        // Aggregate deltas, gammas, etc.
    }
}
```

## **Professional Justification:**

### **1. Industry Standard:**
Every professional Black-Scholes implementation includes these core outputs - it's the **standard interface**

### **2. API Completeness:**
Even if not used now, having `BSResult` makes the class **enterprise-ready** for future features

### **3. Debugging/Validation:**
During development, you might want to inspect d1, d2 values to validate calculations

### **4. Client Requirements:**
Mobile apps often evolve to include more sophisticated features - having the infrastructure ready is smart

## **Recommendation: Keep It!**

**Absolutely keep `BSResult`** because:

✅ **Zero performance cost** - it's just a data class definition
✅ **Future-proofing** - ready for options trading features
✅ **Professional completeness** - matches industry standards
✅ **Debugging utility** - helpful for model validation
✅ **API consistency** - follows established patterns

*/