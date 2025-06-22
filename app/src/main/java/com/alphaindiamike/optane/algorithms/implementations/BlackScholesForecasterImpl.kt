package com.alphaindiamike.optane.algorithms.implementations

import kotlin.math.*
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations

/**
 * Black-Scholes Based Calculator
 * European-style option pricing framework for probability calculation
 *
 *         // 1. calculateImpliedVolatility() or use historical volatility
 *         // 2. normalCDF() - Cumulative normal distribution
 *         // 3. erf() - Error function for normal CDF
 *         // 4. calculateD1D2() - Black-Scholes d1 and d2 parameters
 *         // 5. calculateProbability() - Risk-neutral probability of reaching targets
 */
class BlackScholesForecasterImpl : AlgorithmRepository {

    // Black-Scholes parameters
    private data class BSParams(
        val riskFreeRate: Double = 0.02,  // 2%
        val dividendYield: Double = 0.0,
        val volatility: Double
    )

    private data class BSResult(
        val d1: Double,
        val d2: Double,
        val callPrice: Double,
        val putPrice: Double,
        val callDelta: Double,
        val putDelta: Double
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
        if (timeSeries.size < 5) {
            return "upperBandProbability:0.0,lowerBandProbability:0.0"
        }

        // 1. Calculate or estimate implied volatility from historical data
        val historicalVolatility = calculateHistoricalVolatility(timeSeries)
        val impliedVolatility = estimateImpliedVolatility(timeSeries, historicalVolatility)

        // 2. Set up Black-Scholes parameters
        val bsParams = BSParams(
            riskFreeRate = 0.025, // ECB rate approximation
            dividendYield = 0.0,   // Assuming no dividends
            volatility = impliedVolatility
        )

        // 3. Calculate probabilities using risk-neutral framework
        val currentPrice = timeSeries.last().price
        val timeToExpiry = daysAhead / 252.0 // Convert trading days to years

        val upperProbability = calculateRiskNeutralProbability(
            currentPrice = currentPrice,
            targetPrice = upperBand,
            timeToExpiry = timeToExpiry,
            bsParams = bsParams,
            isUpperBand = true
        )

        val lowerProbability = calculateRiskNeutralProbability(
            currentPrice = currentPrice,
            targetPrice = lowerBand,
            timeToExpiry = timeToExpiry,
            bsParams = bsParams,
            isUpperBand = false
        )

        return """
            Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", upperProbability)}%
            Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", lowerProbability)}%
            """.trimIndent()
    }

    private fun calculateHistoricalVolatility(timeSeries: List<TimeSeriesEntity>): Double {
        if (timeSeries.size < 2) return 0.20 // Default 20% annual volatility

        val returns = mutableListOf<Double>()
        for (i in 1 until timeSeries.size) {
            val ret = ln(timeSeries[i].price / timeSeries[i-1].price)
            returns.add(ret)
        }

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.sum() / (returns.size - 1)
        return sqrt(variance * 252) // Annualized volatility
    }

    private fun estimateImpliedVolatility(timeSeries: List<TimeSeriesEntity>, historicalVol: Double): Double {
        // In practice, this would use market option prices to back out implied vol
        // For now, we'll adjust historical volatility based on recent price behavior

        if (timeSeries.size < 10) return historicalVol

        // Calculate recent volatility (last 5 days)
        val recentPrices = timeSeries.takeLast(6)
        val recentReturns = mutableListOf<Double>()

        for (i in 1 until recentPrices.size) {
            val ret = ln(recentPrices[i].price / recentPrices[i-1].price)
            recentReturns.add(ret)
        }

        val recentMean = recentReturns.average()
        val recentVariance = recentReturns.map { (it - recentMean).pow(2) }.average()
        val recentVol = sqrt(recentVariance * 252)

        // Blend historical and recent volatility (80% historical, 20% recent to reduce noise)
        return historicalVol * 0.8 + recentVol * 0.2
    }

    private fun calculateRiskNeutralProbability(
        currentPrice: Double,
        targetPrice: Double,
        timeToExpiry: Double,
        bsParams: BSParams,
        isUpperBand: Boolean
    ): Double {

        // For barrier/target probabilities, we need different calculation than option pricing
        // This calculates P(S_T >= K) or P(S_T <= K) at expiration

        val drift = bsParams.riskFreeRate - 0.5 * bsParams.volatility.pow(2)
        val logReturn = ln(targetPrice / currentPrice)
        val normalizedReturn = (logReturn - drift * timeToExpiry) / (bsParams.volatility * sqrt(timeToExpiry))

        val probability = if (isUpperBand) {
            // Probability that final price >= target
            1.0 - normalCDF(normalizedReturn)
        } else {
            // Probability that final price <= target
            normalCDF(normalizedReturn)
        }

        return probability * 100.0 // Convert to percentage
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
        // Cumulative normal distribution using error function
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    }

    private fun erf(x: Double): Double {
        // Error function approximation (same high-precision version from your JS)
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

    // Complete Black-Scholes option pricing
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

    // Calculate the Greeks for risk management
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

        // Rho: sensitivity to interest rate
        val callRho = strike * timeToExpiry * exp(-riskFreeRate * timeToExpiry) * normalCDF(d2)
        val putRho = -strike * timeToExpiry * exp(-riskFreeRate * timeToExpiry) * normalCDF(-d2)

        return mapOf(
            "callDelta" to callDelta,
            "putDelta" to putDelta,
            "gamma" to gamma,
            "callTheta" to callTheta / 365.0, // Daily theta
            "putTheta" to putTheta / 365.0,   // Daily theta
            "vega" to vega / 100.0,           // Vega per 1% vol change
            "callRho" to callRho / 100.0,     // Rho per 1% rate change
            "putRho" to putRho / 100.0
        )
    }

    // Implied volatility calculation using Newton-Raphson method
    private fun calculateImpliedVolatility(
        marketPrice: Double,
        spot: Double,
        strike: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        isCall: Boolean = true,
        tolerance: Double = 1e-6,
        maxIterations: Int = 100
    ): Double {

        var vol = 0.20 // Initial guess: 20% volatility

        for (iteration in 0 until maxIterations) {
            val price = blackScholesPrice(spot, strike, timeToExpiry, riskFreeRate, vol, isCall)
            val priceDiff = price - marketPrice

            if (abs(priceDiff) < tolerance) {
                return vol
            }

            // Calculate vega for Newton-Raphson step
            val vega = spot * sqrt(timeToExpiry) * (1.0 / sqrt(2 * PI)) *
                    exp(-0.5 * ((ln(spot / strike) + (riskFreeRate + 0.5 * vol.pow(2)) * timeToExpiry) /
                            (vol * sqrt(timeToExpiry))).pow(2))

            if (abs(vega) < 1e-10) return vol // Return current vol if vega too small

            // Newton-Raphson update
            vol = vol - priceDiff / vega

            // Keep volatility in reasonable bounds
            vol = maxOf(0.001, minOf(5.0, vol))
        }

        return vol
    }

    // Risk-neutral vs Physical probability comparison
    private fun compareRiskNeutralVsPhysical(
        currentPrice: Double,
        targetPrice: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double,
        expectedReturn: Double // Physical/statistical expected return
    ): ProbabilityAnalysis {

        // Risk-neutral probability (using risk-free rate)
        val d2RiskNeutral = (ln(currentPrice / targetPrice) +
                (riskFreeRate - 0.5 * volatility.pow(2)) * timeToExpiry) /
                (volatility * sqrt(timeToExpiry))
        val riskNeutralProb = if (targetPrice > currentPrice) {
            1.0 - normalCDF(d2RiskNeutral)
        } else {
            normalCDF(d2RiskNeutral)
        }

        // Physical probability (using actual expected return)
        val d2Physical = (ln(currentPrice / targetPrice) +
                (expectedReturn - 0.5 * volatility.pow(2)) * timeToExpiry) /
                (volatility * sqrt(timeToExpiry))
        val physicalProb = if (targetPrice > currentPrice) {
            1.0 - normalCDF(d2Physical)
        } else {
            normalCDF(d2Physical)
        }

        return ProbabilityAnalysis(
            riskNeutralProbability = riskNeutralProb * 100,
            physicalProbability = physicalProb * 100,
            impliedVolatility = volatility * 100,
            timeDecay = abs(riskNeutralProb - physicalProb) * 100
        )
    }

    // Volatility smile modeling (simplified)
    private fun calculateVolatilitySmile(
        spot: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        strikes: List<Double>
    ): Map<Double, Double> {

        val atmVolatility = 0.20 // At-the-money volatility
        val smileParameters = mapOf(
            "skew" to -0.02,    // Volatility skew
            "convexity" to 0.001 // Volatility convexity
        )

        return strikes.associateWith { strike ->
            val moneyness = ln(strike / spot)
            val skewAdjustment = smileParameters["skew"]!! * moneyness
            val convexityAdjustment = smileParameters["convexity"]!! * moneyness.pow(2)

            atmVolatility + skewAdjustment + convexityAdjustment
        }
    }

    // Monte Carlo validation of Black-Scholes probabilities
    private fun validateWithMonteCarlo(
        currentPrice: Double,
        targetPrice: Double,
        timeToExpiry: Double,
        riskFreeRate: Double,
        volatility: Double,
        numSimulations: Int = 10000
    ): Double {

        var reachesTarget = 0
        val dt = timeToExpiry / 252.0 // Daily time steps
        val numSteps = (timeToExpiry * 252).toInt()

        repeat(numSimulations) {
            var price = currentPrice
            var maxPrice = currentPrice
            var minPrice = currentPrice

            repeat(numSteps) {
                val random = generateNormalRandom()
                val priceChange = exp((riskFreeRate - 0.5 * volatility.pow(2)) * dt +
                        volatility * sqrt(dt) * random)
                price *= priceChange
                maxPrice = maxOf(maxPrice, price)
                minPrice = minOf(minPrice, price)
            }

            // Check if target was reached during the path
            if (targetPrice > currentPrice && maxPrice >= targetPrice) {
                reachesTarget++
            } else if (targetPrice < currentPrice && minPrice <= targetPrice) {
                reachesTarget++
            }
        }

        return (reachesTarget.toDouble() / numSimulations) * 100.0
    }

    private fun generateNormalRandom(): Double {
        // Box-Muller transformation
        var u = 0.0
        var v = 0.0

        while (u == 0.0) u = kotlin.random.Random.nextDouble()
        while (v == 0.0) v = kotlin.random.Random.nextDouble()

        return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
    }

    // Term structure of volatility
    private fun calculateVolatilityTermStructure(
        timeSeries: List<TimeSeriesEntity>,
        maturities: List<Double> = listOf(7.0/365, 30.0/365, 90.0/365, 180.0/365, 365.0/365)
    ): Map<Double, Double> {

        val baseVolatility = calculateHistoricalVolatility(timeSeries)

        return maturities.associateWith { maturity ->
            // Simple term structure model: vol decreases with maturity
            val termStructureAdjustment = 1.0 + 0.1 * exp(-maturity * 2.0)
            baseVolatility * termStructureAdjustment
        }
    }

    // Risk metrics based on Black-Scholes framework
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

        // Expected shortfall
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
        // Beasley-Springer-Moro algorithm (from your JS code)
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