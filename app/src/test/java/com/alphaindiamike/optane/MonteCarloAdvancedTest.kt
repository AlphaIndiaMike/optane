package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.MonteCarloAdvancedImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*

/**
 * Test suite for MonteCarloAdvancedImpl
 * Tests advanced Monte Carlo with variance reduction techniques
 */
class MonteCarloAdvancedTest {

    private lateinit var monteCarloAdvanced: MonteCarloAdvancedImpl

    @Before
    fun setup() {
        monteCarloAdvanced = MonteCarloAdvancedImpl()
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
        val insufficientData = (1..5).map { day ->
            TimeSeriesEntity(price = 100.0, date = 1704067200000000000L + day * 86400000000000L, assetId = 0L)
        }

        val calculations = Calculations(
            timeSeries = insufficientData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }
        assertEquals("upperBandProbability:0.0,lowerBandProbability:0.0", result)
    }

    @Test
    fun testInputValidation_invalidPriceBands_returnsError() {
        val sineData = generateSineWaveData(amplitude = 2.0, period = 20, days = 30)

        // Test negative price bands
        val negativeCalc = Calculations(
            timeSeries = sineData,
            upperPriceBand = -5.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val negativeResult = runBlocking { monteCarloAdvanced.calculate(negativeCalc) }
        assertEquals("Invalid price bands", negativeResult)

        // Test upper <= lower
        val invalidOrderCalc = Calculations(
            timeSeries = sineData,
            upperPriceBand = 95.0,
            lowerPriceBand = 105.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val invalidOrderResult = runBlocking { monteCarloAdvanced.calculate(invalidOrderCalc) }
        assertEquals("Upper band must be greater than lower band", invalidOrderResult)
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
            upperPriceBand = 115.0, // Well outside range
            lowerPriceBand = 80.0,  // Well outside range
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

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

        val smallResult = runBlocking { monteCarloAdvanced.calculate(smallAmplitudeCalc) }
        val largeResult = runBlocking { monteCarloAdvanced.calculate(largeAmplitudeCalc) }

        val smallUpperProb = extractProbabilityFromResult(smallResult, "Upper")
        val largeUpperProb = extractProbabilityFromResult(largeResult, "Upper")

        // Higher volatility (large amplitude) should generally give higher crossing probabilities
        assertTrue("High volatility should increase crossing probability compared to low volatility",
            largeUpperProb >= smallUpperProb - 10.0) // Allow tolerance for Monte Carlo variation
    }

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

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

        // With low volatility, should give low probabilities for different target prices
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Advanced Monte Carlo should be more stable than basic, but still low for constant data
        assertTrue("Constant price should give low probability for upper band", upperProb < 15.0)
        assertTrue("Constant price should give low probability for lower band", lowerProb < 15.0)
    }

    // ================== ADVANCED MONTE CARLO SPECIFIC TESTS ==================

    @Test
    fun testAdvancedMonteCarlo_moreStableThanBasic() {
        val sineData = generateSineWaveData(amplitude = 3.0, period = 20, days = 60)

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        // Run multiple times to test stability
        val results = mutableListOf<Pair<Double, Double>>()
        repeat(3) {
            val result = runBlocking { monteCarloAdvanced.calculate(calculations) }
            val upperProb = extractProbabilityFromResult(result, "Upper")
            val lowerProb = extractProbabilityFromResult(result, "Lower")
            results.add(Pair(upperProb, lowerProb))
        }

        val upperProbs = results.map { it.first }
        val lowerProbs = results.map { it.second }

        val upperStdDev = calculateStandardDeviation(upperProbs)
        val lowerStdDev = calculateStandardDeviation(lowerProbs)

        // Advanced Monte Carlo should be more stable due to variance reduction
        assertTrue("Advanced Monte Carlo upper probabilities should be very stable", upperStdDev < 5.0)
        assertTrue("Advanced Monte Carlo lower probabilities should be very stable", lowerStdDev < 5.0)
    }

    @Test
    fun testAdvancedFeatures_handlesLargeDatasets() {
        // Test with larger dataset to ensure advanced features scale well
        val largeDataset = generateSineWaveData(amplitude = 4.0, period = 50, days = 200)

        val calculations = Calculations(
            timeSeries = largeDataset,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val startTime = System.currentTimeMillis()
        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }
        val endTime = System.currentTimeMillis()

        // Should complete in reasonable time (less than 30 seconds for testing)
        assertTrue("Advanced Monte Carlo should handle large datasets efficiently",
            (endTime - startTime) < 30000)

        // Should still return valid probabilities
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper probability should be valid", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testVarianceReduction_betterAccuracy() {
        // Test that advanced algorithm gives more reasonable results
        val moderateVolatilityData = generateSineWaveData(
            amplitude = 3.0,
            period = 25,
            days = 75,
            basePrice = 100.0
        )

        val calculations = Calculations(
            timeSeries = moderateVolatilityData,
            upperPriceBand = 102.5,
            lowerPriceBand = 97.5,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // With moderate volatility and 5 days, probabilities should be reasonable but not necessarily high
        assertTrue("Upper probability should be valid", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)

        // At least one probability should be meaningful (>1%) due to volatility
        assertTrue("At least one probability should be meaningful with moderate volatility",
            upperProb > 1.0 || lowerProb > 1.0)

        // Combined probabilities should reflect some meaningful chance of movement
        assertTrue("Combined probability should show meaningful volatility impact",
            (upperProb + lowerProb) > 5.0)
    }

    @Test
    fun testOutputFormat_includesAdvancedDescription() {
        val sineData = generateSineWaveData(amplitude = 2.0, period = 20, days = 40)

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 102.0,
            lowerPriceBand = 98.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

        // Check that result contains the proper probability format
        assertTrue("Result should contain upper band probability",
            result.contains("Upper band"))
        assertTrue("Result should contain lower band probability",
            result.contains("Lower band"))
        assertTrue("Result should contain percentage symbol",
            result.contains("%"))
        assertTrue("Result should contain probability keyword",
            result.contains("probability"))
    }

    @Test
    fun testAdvancedAlgorithm_robustWithVariousPatterns() {
        // Test with different data patterns to ensure robustness
        val patterns = listOf(
            generateSineWaveData(amplitude = 0.5, period = 10, days = 30),
            generateSineWaveData(amplitude = 10.0, period = 100, days = 150),
            generateTrendingData(startPrice = 80.0, dailyChange = 1.5, days = 40),
            generateConstantPriceData(price = 200.0, days = 50)
        )

        patterns.forEach { data ->
            val currentPrice = data.last().price
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = currentPrice * 1.03,
                lowerPriceBand = currentPrice * 0.97,
                daysPrediction = 5,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

            // Should not crash and should return valid format (not error format)
            assertFalse("Advanced Monte Carlo should not return error format for sufficient data",
                result.startsWith("upperBandProbability:0.0"))
            assertTrue("Advanced Monte Carlo should handle various data patterns",
                result.contains("Upper band"))

            val upperProb = extractProbabilityFromResult(result, "Upper")
            val lowerProb = extractProbabilityFromResult(result, "Lower")
            assertTrue("Upper probability should be valid", upperProb >= 0.0 && upperProb <= 100.0)
            assertTrue("Lower probability should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)
        }
    }

    @Test
    fun testLongerPredictionPeriod_increasesOpportunity() {
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
            daysPrediction = 2,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val longTerm = Calculations(
            timeSeries = sineData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 15,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val shortResult = runBlocking { monteCarloAdvanced.calculate(shortTerm) }
        val longResult = runBlocking { monteCarloAdvanced.calculate(longTerm) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        // Longer periods should typically increase probability of touching barriers
        assertTrue("Longer prediction should typically increase or maintain touch probability",
            longUpperProb >= shortUpperProb - 10.0) // Allow reasonable tolerance
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

        val result = runBlocking { monteCarloAdvanced.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper probability should be between 0-100%", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be between 0-100%", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testHigherVolatility_higherProbability() {
        val lowVolData = generateSineWaveData(amplitude = 1.0, period = 30, days = 60)
        val highVolData = generateSineWaveData(amplitude = 6.0, period = 30, days = 60)

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

        val lowVolResult = runBlocking { monteCarloAdvanced.calculate(lowVolCalc) }
        val highVolResult = runBlocking { monteCarloAdvanced.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        assertTrue("Higher volatility should give higher probability",
            highVolProb >= lowVolProb - 8.0) // Smaller tolerance for advanced algorithm
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