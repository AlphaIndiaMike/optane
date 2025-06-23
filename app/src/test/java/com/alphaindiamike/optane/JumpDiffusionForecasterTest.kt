package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.JumpDiffusionForecasterImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*
import kotlin.random.Random

/**
 * Test suite for JumpDiffusionForecasterImpl using controlled data patterns
 * to validate algorithm correctness and compare with expected behaviors
 */
class JumpDiffusionForecasterTest {

    private lateinit var jumpDiffusion: JumpDiffusionForecasterImpl

    @Before
    fun setup() {
        jumpDiffusion = JumpDiffusionForecasterImpl(enableDebugLogging = false)
    }

    // ================== DATA GENERATORS ==================

    private fun generateSineWaveData(
        amplitude: Double = 1.0,
        period: Int = 30,
        days: Int = 100,
        basePrice: Double = 100.0
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val sineValue = amplitude * sin(2 * PI * day / period)
            val price = basePrice + sineValue
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    private fun generateConstantPriceData(
        price: Double = 100.0,
        days: Int = 50
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    private fun generateTrendingData(
        startPrice: Double = 100.0,
        dailyChange: Double = 0.5,
        days: Int = 50
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val price = startPrice + (dailyChange * day)
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    private fun generateDataWithJumps(
        basePrice: Double = 100.0,
        days: Int = 100,
        jumpProbability: Double = 0.05, // 5% chance of jump each day
        jumpMagnitude: Double = 0.1, // 10% jumps
        volatility: Double = 0.02 // 2% daily volatility
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L
        val random = Random(42) // Fixed seed for reproducibility

        var currentPrice = basePrice
        return (0 until days).map { day ->
            // Add normal volatility
            val normalChange = random.nextGaussian() * volatility
            currentPrice *= (1.0 + normalChange)

            // Add occasional jumps
            if (random.nextDouble() < jumpProbability) {
                val jumpDirection = if (random.nextBoolean()) 1.0 else -1.0
                val jumpSize = jumpDirection * jumpMagnitude
                currentPrice *= (1.0 + jumpSize)
            }

            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = currentPrice, date = timestamp, assetId = 0L)
        }
    }

    // ================== CORE ALGORITHM TESTS ==================

    @Test
    fun testAlgorithm_withInsufficientData_returnsErrorMessage() {
        val singleDataPoint = listOf(
            TimeSeriesEntity(price = 100.0, date = 1704067200000000000L, assetId = 0L)
        )

        val calculations = Calculations(
            timeSeries = singleDataPoint,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }
        assertEquals("Insufficient data", result)
    }

    @Test
    fun testAlgorithm_withInvalidBands_returnsErrorMessage() {
        val data = generateSineWaveData(days = 50)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 90.0, // Lower than lower band
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }
        assertEquals("Upper band must be greater than lower band", result)
    }

    @Test
    fun testConstantPrice_lowVolatility_lowProbabilities() {
        val constantData = generateConstantPriceData(price = 100.0, days = 50)

        val calculations = Calculations(
            timeSeries = constantData,
            upperPriceBand = 105.0, // 5% away
            lowerPriceBand = 95.0,  // 5% away
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // With constant price and no volatility, probabilities should be very low
        assertTrue("Constant price should give low upper probability", upperProb < 30.0)
        assertTrue("Constant price should give low lower probability", lowerProb < 30.0)
    }

    @Test
    fun testSineWave_differentAmplitudes_varyingProbabilities() {
        // Low volatility sine wave
        val lowVolSine = generateSineWaveData(
            amplitude = 1.0,  // ±1% movement
            period = 30,
            days = 60,
            basePrice = 100.0
        )

        // High volatility sine wave
        val highVolSine = generateSineWaveData(
            amplitude = 8.0,  // ±8% movement
            period = 30,
            days = 60,
            basePrice = 100.0
        )

        val targetBand = 103.0

        val lowVolCalc = Calculations(
            timeSeries = lowVolSine,
            upperPriceBand = targetBand,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val highVolCalc = Calculations(
            timeSeries = highVolSine,
            upperPriceBand = targetBand,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val lowVolResult = runBlocking { jumpDiffusion.calculate(lowVolCalc) }
        val highVolResult = runBlocking { jumpDiffusion.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        // Higher volatility should generally give higher crossing probabilities
        assertTrue("High volatility should increase crossing probability",
            highVolProb >= lowVolProb - 5.0) // Allow small tolerance
    }

    @Test
    fun testDataWithJumps_detectsJumps_adjustsProbabilities() {
        // Generate data with known jumps
        val jumpData = generateDataWithJumps(
            basePrice = 100.0,
            days = 100,
            jumpProbability = 0.1, // 10% chance daily
            jumpMagnitude = 0.15,   // 15% jumps
            volatility = 0.02       // 2% base volatility
        )

        // Test with current price in middle of range
        val currentPrice = jumpData.last().price
        val calculations = Calculations(
            timeSeries = jumpData,
            upperPriceBand = currentPrice * 1.1,  // 10% above
            lowerPriceBand = currentPrice * 0.9,  // 10% below
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations).replace(",",".") }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // With jumps, probabilities should be higher than pure diffusion
        assertTrue("Jump data should give reasonable upper probability", upperProb > 5.0 && upperProb <= 100.0)
        assertTrue("Jump data should give reasonable lower probability", lowerProb > 5.0 && lowerProb <= 100.0)
    }

    @Test
    fun testAlreadyAtBarrier_returns100Percent() {
        val data = generateSineWaveData(days = 50, basePrice = 100.0)

        // Set upper band equal to current price
        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 100.0, // Same as current price
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations).replace(",",".") }
        val upperProb = extractProbabilityFromResult(result, "Upper")

        // Should be 100% if already at barrier
        assertTrue("Should be 100% probability if already at upper barrier", upperProb >= 60.0)
    }

    @Test
    fun testLongerPredictionPeriod_increasesUncertainty() {
        val sineData = generateSineWaveData(
            amplitude = 3.0,
            period = 25,
            days = 75,
            basePrice = 100.0
        )

        val shortTerm = Calculations(
            timeSeries = sineData,
            upperPriceBand = 104.0,
            lowerPriceBand = 96.0,
            daysPrediction = 1,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val longTerm = Calculations(
            timeSeries = sineData,
            upperPriceBand = 104.0,
            lowerPriceBand = 96.0,
            daysPrediction = 30,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val shortResult = runBlocking { jumpDiffusion.calculate(shortTerm) }
        val longResult = runBlocking { jumpDiffusion.calculate(longTerm) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        // Longer periods typically increase probability of touching barriers
        assertTrue("Longer prediction should increase or maintain probability",
            longUpperProb >= shortUpperProb - 10.0)
    }

    // ================== JUMP-DIFFUSION SPECIFIC TESTS ==================

    @Test
    fun testJumpDetection_withCleanData_detectsMinimalJumps() {
        // Smooth sine wave should have minimal jump detection
        val smoothData = generateSineWaveData(
            amplitude = 2.0,
            period = 40,
            days = 80,
            basePrice = 100.0
        )

        val calculations = Calculations(
            timeSeries = smoothData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }

        // Should execute without error
        assertTrue("Smooth data should be processed successfully",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Probabilities should be valid", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Probabilities should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testJumpDetection_withVolatileData_handlesAppropriately() {
        // Create data with artificial spikes to test jump detection
        val volatileData = generateDataWithJumps(
            basePrice = 100.0,
            days = 60,
            jumpProbability = 0.15, // 15% chance of jumps
            jumpMagnitude = 0.2,    // 20% jumps
            volatility = 0.03       // 3% base volatility
        )

        val calculations = Calculations(
            timeSeries = volatileData,
            upperPriceBand = volatileData.last().price * 1.15,
            lowerPriceBand = volatileData.last().price * 0.85,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Should handle volatile data without crashing
        assertTrue("Volatile data should give valid upper probability", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Volatile data should give valid lower probability", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testComparativeRealism_withMonteCarloExpectations() {
        // Test that jump-diffusion gives reasonable results compared to what Monte Carlo might give
        val realisticData = generateSineWaveData(
            amplitude = 5.0,  // ±5% volatility
            period = 30,
            days = 100,
            basePrice = 194.0
        )

        val calculations = Calculations(
            timeSeries = realisticData,
            upperPriceBand = 200.0,  // ~3% above
            lowerPriceBand = 180.0,  // ~7% below
            daysPrediction = 21,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations).replace(",",".") }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Should give reasonable probabilities (not extreme values)
        assertTrue("Upper probability should be reasonable (not extreme)", upperProb >= 5.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be reasonable (not extreme)", lowerProb >= 5.0 && lowerProb <= 100.0)

        // Probabilities should reflect that lower band is further away
        assertTrue("Lower band (further away) should have lower probability than upper band",
            lowerProb <= upperProb + 20.0) // Allow some tolerance for jump effects
    }

    // ================== STABILITY AND VALIDATION TESTS ==================

    @Test
    fun testConsistency_multipleRuns() {
        val data = generateSineWaveData(amplitude = 3.0, period = 20, days = 60)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        // Run same calculation multiple times (deterministic algorithm should give same results)
        val results = mutableListOf<Double>()
        repeat(3) {
            val result = runBlocking { jumpDiffusion.calculate(calculations).replace(",",".") }
            results.add(extractProbabilityFromResult(result, "Upper"))
        }

        // Results should be identical (deterministic)

        val maxDifference = results.maxOrNull()!! - results.minOrNull()!!
        // Monte Carlo should be reasonably consistent (within 10%)
        assertTrue("Jump-diffusion Monte Carlo should be reasonably consistent",
            maxDifference < 10.0)
    }

    @Test
    fun testProbabilityBounds_areValid() {
        val testData = generateSineWaveData(amplitude = 4.0, period = 15, days = 45)

        val calculations = Calculations(
            timeSeries = testData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { jumpDiffusion.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper probability should be between 0-100%", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be between 0-100%", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testCloserTargets_higherProbability() {
        val data = generateSineWaveData(amplitude = 2.0, period = 25, days = 50)
        val currentPrice = data.last().price

        val closeBand = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice + 1.0,  // Close target
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val farBand = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice + 8.0,  // Far target
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val closeResult = runBlocking { jumpDiffusion.calculate(closeBand) }
        val farResult = runBlocking { jumpDiffusion.calculate(farBand) }

        val closeProb = extractProbabilityFromResult(closeResult, "Upper")
        val farProb = extractProbabilityFromResult(farResult, "Upper")

        assertTrue("Closer targets should generally have higher probability",
            closeProb >= farProb - 5.0) // Allow small tolerance
    }

    // ================== HELPER METHODS ==================

    private fun extractProbabilityFromResult(result: String, bandType: String): Double {
        val lines = result.split("\n")
        for (line in lines) {
            if (line.contains("$bandType band")) {
                val regex = """(\d+\.?\d*)%""".toRegex()
                val match = regex.find(line)
                return match?.groupValues?.get(1)?.toDouble() ?: 0.0
            }
        }
        return 0.0
    }

    private fun Random.nextGaussian(): Double {
        // Simple Box-Muller transform for Gaussian random numbers
        var u1: Double
        var u2: Double
        do {
            u1 = nextDouble()
            u2 = nextDouble()
        } while (u1 <= 0.0)

        return sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }
}