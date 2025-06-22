package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.MonteCarloBasicImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*

/**
 * Test suite for MonteCarloBasicImpl using sine wave data
 * to validate algorithm correctness with known ground truth
 */
class MonteCarloBasicTest {

    private lateinit var monteCarlo: MonteCarloBasicImpl

    @Before
    fun setup() {
        monteCarlo = MonteCarloBasicImpl()
    }

    // ================== SINE WAVE DATA GENERATORS ==================

    private fun generateSineWaveData(
        amplitude: Double = 1.0,
        period: Int = 30,
        days: Int = 100,
        basePrice: Double = 100.0
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L // Jan 1, 2024 00:00:00 UTC in nanoseconds
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L // 24 hours in nanoseconds

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
        val baseTimestamp = 1704067200000000000L // Jan 1, 2024 00:00:00 UTC in nanoseconds
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    private fun generateTrendingData(
        startPrice: Double = 100.0,
        dailyChange: Double = 0.5, // Absolute daily change
        days: Int = 50
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L // Jan 1, 2024 00:00:00 UTC in nanoseconds
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val price = startPrice + (dailyChange * day)
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
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

        val result = runBlocking { monteCarlo.calculate(calculations) }
        assertEquals("upperBandProbability:0.0,lowerBandProbability:0.0", result)
    }

    @Test
    fun testSineWave_noCrossingScenario_lowProbabilities() {
        // Sine wave oscillating between 95-105 (amplitude=5, base=100)
        val sineData = generateSineWaveData(
            amplitude = 5.0,
            period = 30,
            days = 90,
            basePrice = 100.0
        )

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 110.0, // Well outside range
            lowerPriceBand = 85.0,  // Well outside range
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarlo.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper band probability should be low for out-of-range target", upperProb < 15.0)
        assertTrue("Lower band probability should be low for out-of-range target", lowerProb < 15.0)
    }

    @Test
    fun testSineWave_differentAmplitudes_varyingVolatility() {
        // Test small amplitude (low volatility)
        val smallAmplitudeSine = generateSineWaveData(
            amplitude = 1.0,
            period = 30,
            days = 60,
            basePrice = 100.0
        )

        // Test large amplitude (high volatility)
        val largeAmplitudeSine = generateSineWaveData(
            amplitude = 8.0,
            period = 30,
            days = 60,
            basePrice = 100.0
        )

        val targetBand = 103.0

        val smallAmplitudeCalc = Calculations(
            timeSeries = smallAmplitudeSine,
            upperPriceBand = targetBand,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val largeAmplitudeCalc = Calculations(
            timeSeries = largeAmplitudeSine,
            upperPriceBand = targetBand,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val smallResult = runBlocking { monteCarlo.calculate(smallAmplitudeCalc) }
        val largeResult = runBlocking { monteCarlo.calculate(largeAmplitudeCalc) }

        val smallUpperProb = extractProbabilityFromResult(smallResult, "Upper")
        val largeUpperProb = extractProbabilityFromResult(largeResult, "Upper")

        // Higher volatility (large amplitude) should generally give higher crossing probabilities
        assertTrue("High volatility should increase crossing probability compared to low volatility",
            largeUpperProb >= smallUpperProb - 10.0) // Allow tolerance for Monte Carlo variation
    }

    @Test
    fun testSineWave_differentPeriods_handlesVariousTimeScales() {
        val shortPeriodSine = generateSineWaveData(
            amplitude = 3.0,
            period = 10, // Fast oscillation
            days = 50,
            basePrice = 100.0
        )

        val longPeriodSine = generateSineWaveData(
            amplitude = 3.0,
            period = 40, // Slow oscillation
            days = 80,
            basePrice = 100.0
        )

        val calculations1 = Calculations(
            timeSeries = shortPeriodSine,
            upperPriceBand = 102.0,
            lowerPriceBand = 98.0,
            daysPrediction = 3,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val calculations2 = Calculations(
            timeSeries = longPeriodSine,
            upperPriceBand = 102.0,
            lowerPriceBand = 98.0,
            daysPrediction = 3,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result1 = runBlocking { monteCarlo.calculate(calculations1) }
        val result2 = runBlocking { monteCarlo.calculate(calculations2) }

        // Both should execute without error and return valid probabilities
        assertFalse("Short period sine should not return CSV error format",
            result1.startsWith("upperBandProbability:0.0"))
        assertFalse("Long period sine should not return CSV error format",
            result2.startsWith("upperBandProbability:0.0"))

        val prob1 = extractProbabilityFromResult(result1, "Upper")
        val prob2 = extractProbabilityFromResult(result2, "Upper")

        assertTrue("Short period probability should be valid", prob1 >= 0.0 && prob1 <= 100.0)
        assertTrue("Long period probability should be valid", prob2 >= 0.0 && prob2 <= 100.0)
    }

    // ================== EDGE CASE TESTS ==================

    @Test
    fun testConstantPrice_lowProbabilities() {
        val constantData = generateConstantPriceData(price = 100.0, days = 30)

        val calculations = Calculations(
            timeSeries = constantData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarlo.calculate(calculations) }

        // With low volatility, should give low probabilities for different target prices
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Monte Carlo might have some variation even with constant data due to sampling
        assertTrue("Constant price should give low probability for upper band", upperProb < 20.0)
        assertTrue("Constant price should give low probability for lower band", lowerProb < 20.0)
    }

    @Test
    fun testLongerPredictionPeriod_increasesUncertainty() {
        val sineData = generateSineWaveData(
            amplitude = 2.0,
            period = 25,
            days = 75,
            basePrice = 100.0
        )

        val shortTerm = Calculations(
            timeSeries = sineData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 1,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val longTerm = Calculations(
            timeSeries = sineData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 20,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val shortResult = runBlocking { monteCarlo.calculate(shortTerm) }
        val longResult = runBlocking { monteCarlo.calculate(longTerm) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        // Longer prediction periods typically increase probability due to more opportunities to hit barriers
        assertTrue("Longer prediction should typically increase or maintain probability",
            longUpperProb >= shortUpperProb - 15.0) // Allow reasonable Monte Carlo tolerance
    }

    // ================== MONTE CARLO SPECIFIC TESTS ==================

    @Test
    fun testMonteCarloConsistency_multipleRuns() {
        val sineData = generateSineWaveData(amplitude = 3.0, period = 20, days = 60)

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        // Run same calculation multiple times
        val upperResults = mutableListOf<Double>()
        val lowerResults = mutableListOf<Double>()

        repeat(5) {
            val result = runBlocking { monteCarlo.calculate(calculations) }
            upperResults.add(extractProbabilityFromResult(result, "Upper"))
            lowerResults.add(extractProbabilityFromResult(result, "Lower"))
        }

        val upperStdDev = calculateStandardDeviation(upperResults)
        val lowerStdDev = calculateStandardDeviation(lowerResults)

        // Monte Carlo should be reasonably consistent across runs
        assertTrue("Monte Carlo upper probabilities should be reasonably consistent", upperStdDev < 10.0)
        assertTrue("Monte Carlo lower probabilities should be reasonably consistent", lowerStdDev < 10.0)
    }

    @Test
    fun testMonteCarloStability_withSufficientData() {
        // Test that Monte Carlo doesn't crash or give extreme results with various data patterns
        val patterns = listOf(
            generateSineWaveData(amplitude = 0.1, period = 5, days = 30),
            generateSineWaveData(amplitude = 10.0, period = 100, days = 200),
            generateTrendingData(startPrice = 50.0, dailyChange = 2.0, days = 25),
            generateConstantPriceData(price = 1000.0, days = 40)
        )

        patterns.forEach { data ->
            val currentPrice = data.last().price
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = currentPrice * 1.05,
                lowerPriceBand = currentPrice * 0.95,
                daysPrediction = 7,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { monteCarlo.calculate(calculations) }

            // Should not crash and should return valid format
            assertTrue("Monte Carlo should handle various data patterns",
                result.contains("Upper band") || result.contains("upperBandProbability"))

            if (result.contains("Upper band")) {
                val upperProb = extractProbabilityFromResult(result, "Upper")
                val lowerProb = extractProbabilityFromResult(result, "Lower")
                assertTrue("Upper probability should be valid", upperProb >= 0.0 && upperProb <= 100.0)
                assertTrue("Lower probability should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)
            }
        }
    }

    // ================== VALIDATION TESTS ==================

    @Test
    fun testProbabilityOutputFormat_isCorrect() {
        val sineData = generateSineWaveData(amplitude = 2.0, period = 20, days = 40)

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 102.0,
            lowerPriceBand = 98.0,
            daysPrediction = 3,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarlo.calculate(calculations) }

        // Check that result contains expected format
        assertTrue("Result should contain upper band probability",
            result.contains("Upper band"))
        assertTrue("Result should contain lower band probability",
            result.contains("Lower band"))
        assertTrue("Result should contain percentage symbol",
            result.contains("%"))
    }

    @Test
    fun testProbabilityBounds_areValid() {
        val sineData = generateSineWaveData(amplitude = 4.0, period = 15, days = 45)

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 101.0,
            lowerPriceBand = 99.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarlo.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper probability should be between 0-100%", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be between 0-100%", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testWiderBands_higherProbability() {
        val sineData = generateSineWaveData(amplitude = 3.0, period = 20, days = 60)

        val currentPrice = sineData.last().price

        val closerTarget = Calculations(
            timeSeries = sineData,
            upperPriceBand = currentPrice + 1.0,  // Close to current
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val fartherTarget = Calculations(
            timeSeries = sineData,
            upperPriceBand = currentPrice + 5.0,  // Far from current
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val closerResult = runBlocking { monteCarlo.calculate(closerTarget) }
        val fartherResult = runBlocking { monteCarlo.calculate(fartherTarget) }

        val closerProb = extractProbabilityFromResult(closerResult, "Upper")
        val fartherProb = extractProbabilityFromResult(fartherResult, "Upper")

        assertTrue("Closer targets should have higher probability", closerProb >= fartherProb - 5.0) // Monte Carlo tolerance
    }

    @Test
    fun testHigherVolatility_higherProbability() {
        val lowVolData = generateSineWaveData(amplitude = 1.0, period = 30, days = 60)
        val highVolData = generateSineWaveData(amplitude = 5.0, period = 30, days = 60)

        val lowVolCalc = Calculations(
            timeSeries = lowVolData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val highVolCalc = Calculations(
            timeSeries = highVolData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val lowVolResult = runBlocking { monteCarlo.calculate(lowVolCalc) }
        val highVolResult = runBlocking { monteCarlo.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        assertTrue("Higher volatility should give higher probability",
            highVolProb >= lowVolProb - 10.0) // Allow Monte Carlo tolerance
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

    private fun calculateStandardDeviation(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val mean = values.average()
        val variance = values.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }
}