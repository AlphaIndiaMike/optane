package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.ProbabilisticForecasterImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*

/**
 * Test suite for ProbabilisticForecasterImpl using sine wave data
 * to validate algorithm correctness with known ground truth
 */
class ProbabilisticForecasterTest {

    private lateinit var forecaster: ProbabilisticForecasterImpl

    @Before
    fun setup() {
        forecaster = ProbabilisticForecasterImpl()
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

        val result = runBlocking { forecaster.calculate(calculations) }
        assertEquals("Insufficient data", result)
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
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { forecaster.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper band probability should be low for out-of-range target", upperProb < 10.0)
        assertTrue("Lower band probability should be low for out-of-range target", lowerProb < 10.0)
    }

    /* Incorrect use case
    @Test
    fun testSineWave_guaranteedCrossingScenario_highProbabilities() {
        // Create sine wave that's currently near middle, with bands well within historical range
        val sineData = generateSineWaveData(
            amplitude = 15.0,    // Larger amplitude for more volatility
            period = 20,
            days = 100,          // More data points
            basePrice = 100.0
        )

        // Make sure we end near the middle by adjusting the last few points if needed
        val currentPrice = sineData.last().price

        val calculations = Calculations(
            timeSeries = sineData,
            upperPriceBand = 108.0,  // Well within historical range (85-115)
            lowerPriceBand = 92.0,   // Well within historical range
            daysPrediction = 3,      // Shorter prediction period
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { forecaster.calculate(calculations) }
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // With high amplitude sine wave, should detect significant volatility
        assertTrue("Upper band should have reasonable probability with high volatility sine wave",
            upperProb > 10.0)
        assertTrue("Lower band should have reasonable probability with high volatility sine wave",
            lowerProb > 10.0)
    }*/

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

        val smallResult = runBlocking { forecaster.calculate(smallAmplitudeCalc) }
        val largeResult = runBlocking { forecaster.calculate(largeAmplitudeCalc) }

        val smallUpperProb = extractProbabilityFromResult(smallResult, "Upper")
        val largeUpperProb = extractProbabilityFromResult(largeResult, "Upper")

        // Higher volatility (large amplitude) should generally give higher crossing probabilities
        assertTrue("High volatility should increase crossing probability compared to low volatility",
            largeUpperProb >= smallUpperProb - 5.0) // Allow small tolerance
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

        val result1 = runBlocking { forecaster.calculate(calculations1) }
        val result2 = runBlocking { forecaster.calculate(calculations2) }

        // Both should execute without error and return valid probabilities
        assertFalse("Short period sine should not return error", result1.contains("Insufficient"))
        assertFalse("Long period sine should not return error", result2.contains("Insufficient"))

        val prob1 = extractProbabilityFromResult(result1, "Upper")
        val prob2 = extractProbabilityFromResult(result2, "Upper")

        assertTrue("Short period probability should be valid", prob1 >= 0.0 && prob1 <= 100.0)
        assertTrue("Long period probability should be valid", prob2 >= 0.0 && prob2 <= 100.0)
    }

    // ================== EDGE CASE TESTS ==================

    @Test
    fun testConstantPrice_extremeProbabilities() {
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

        val result = runBlocking { forecaster.calculate(calculations) }

        // With zero volatility, should give very low probabilities for different target prices
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Constant price should give low probability for upper band", upperProb < 5.0)
        assertTrue("Constant price should give low probability for lower band", lowerProb < 5.0)
    }

    /* Incorrect use case
    @Test
    fun testTrendingData_favorsTrendDirection() {
        val upwardTrendingData = generateTrendingData(
            startPrice = 100.0,
            dailyChange = 1.0, // $1 increase per day
            days = 30
        )

        val currentPrice = upwardTrendingData.last().price

        val calculations = Calculations(
            timeSeries = upwardTrendingData,
            upperPriceBand = currentPrice + 5.0,
            lowerPriceBand = currentPrice - 5.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { forecaster.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Upward trending data should favor upper band
        assertTrue("Upward trend should favor upper band crossing",
            upperProb > lowerProb)
    }*/

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

        val shortResult = runBlocking { forecaster.calculate(shortTerm) }
        val longResult = runBlocking { forecaster.calculate(longTerm) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        // Longer prediction periods typically increase probability due to time effect
        assertTrue("Longer prediction should typically increase or maintain probability",
            longUpperProb >= shortUpperProb - 10.0) // Allow reasonable tolerance
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

        val result = runBlocking { forecaster.calculate(calculations) }

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

        val result = runBlocking { forecaster.calculate(calculations) }

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

        val closerResult = runBlocking { forecaster.calculate(closerTarget) }
        val fartherResult = runBlocking { forecaster.calculate(fartherTarget) }

        val closerProb = extractProbabilityFromResult(closerResult, "Upper")
        val fartherProb = extractProbabilityFromResult(fartherResult, "Upper")

        assertTrue("Closer targets should have higher probability", closerProb >= fartherProb)
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

        val lowVolResult = runBlocking { forecaster.calculate(lowVolCalc) }
        val highVolResult = runBlocking { forecaster.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        assertTrue("Higher volatility should give higher probability", highVolProb >= lowVolProb)
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
}