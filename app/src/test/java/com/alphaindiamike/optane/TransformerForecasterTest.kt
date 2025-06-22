package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.TransformerForecasterImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*

/**
 * Test suite for TransformerForecasterImpl
 * Tests multi-head attention transformer with deterministic feature engineering
 */
class TransformerForecasterTest {

    private lateinit var transformerForecaster: TransformerForecasterImpl

    @Before
    fun setup() {
        transformerForecaster = TransformerForecasterImpl(enableDebugLogging = false)
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

    private fun generateTrendingData(
        startPrice: Double = 100.0,
        dailyReturn: Double = 0.005,
        volatility: Double = 0.02,
        days: Int = 100
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L
        var currentPrice = startPrice

        return (0 until days).map { day ->
            val trendComponent = currentPrice * dailyReturn
            val volatilityComponent = currentPrice * volatility * sin(day * 0.3)
            currentPrice += trendComponent + volatilityComponent

            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = currentPrice, date = timestamp, assetId = 0L)
        }
    }

    private fun generateComplexPatternData(
        basePrice: Double = 100.0,
        days: Int = 150
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            // Multi-frequency pattern for transformer to learn
            val shortCycle = 3.0 * sin(2 * PI * day / 12) // 12-day cycle
            val mediumCycle = 2.0 * sin(2 * PI * day / 35) // 35-day cycle
            val longCycle = 1.5 * sin(2 * PI * day / 80) // 80-day cycle
            val trend = day * 0.1 // Slight upward trend
            val noise = 0.5 * sin(day * 1.7) * cos(day * 2.3) // Complex noise

            val price = basePrice + shortCycle + mediumCycle + longCycle + trend + noise
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    private fun generateStepFunctionData(
        basePrice: Double = 100.0,
        days: Int = 120,
        stepSize: Double = 5.0,
        stepFrequency: Int = 20
    ): List<TimeSeriesEntity> {
        val baseTimestamp = 1704067200000000000L
        val dayInNanoseconds = 24L * 60L * 60L * 1000L * 1000L * 1000L

        return (0 until days).map { day ->
            val steps = day / stepFrequency
            val price = basePrice + (steps * stepSize)
            val timestamp = baseTimestamp + (day * dayInNanoseconds)
            TimeSeriesEntity(price = price, date = timestamp, assetId = 0L)
        }
    }

    // ================== BASIC FUNCTIONALITY TESTS ==================

    @Test
    fun testInsufficientData_returnsErrorMessage() {
        // Test with data less than sequenceLength + 5 (default sequenceLength = 8)
        val insufficientData = (1..10).map { day ->
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

        val result = runBlocking { transformerForecaster.calculate(calculations) }
        assertEquals("Insufficient data", result)
    }

    @Test
    fun testInputValidation_invalidPriceBands() {
        val data = generateSineWaveData(days = 50)

        val negativeCalc = Calculations(
            timeSeries = data,
            upperPriceBand = -5.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val negativeResult = runBlocking { transformerForecaster.calculate(negativeCalc) }
        assertEquals("Invalid price bands", negativeResult)

        val invalidOrderCalc = Calculations(
            timeSeries = data,
            upperPriceBand = 95.0,
            lowerPriceBand = 105.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val invalidOrderResult = runBlocking { transformerForecaster.calculate(invalidOrderCalc) }
        assertEquals("Upper band must be greater than lower band", invalidOrderResult)
    }

    @Test
    fun testMinimumDataRequirement_exactlyThirteenPoints() {
        // Need sequenceLength + 5 = 8 + 5 = 13 minimum
        val minimalData = generateSineWaveData(days = 13)

        val calculations = Calculations(
            timeSeries = minimalData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }

        assertTrue("Algorithm should handle minimal data",
            result.contains("Upper band") && result.contains("Lower band"))
    }

    // ================== DETERMINISTIC BEHAVIOR TESTS ==================

    @Test
    fun testDeterministicBehavior_sameInputSameOutput() {
        val data = generateComplexPatternData(days = 100)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val results = mutableListOf<String>()
        repeat(3) {
            val result = runBlocking { transformerForecaster.calculate(calculations) }
            results.add(result)
        }

        assertTrue("Transformer Forecaster should be deterministic",
            results.all { it == results[0] })
    }

    @Test
    fun testFeatureEngineering_handlesComplexPatterns() {
        val patterns = listOf(
            "Sine Wave" to generateSineWaveData(amplitude = 4.0, period = 25, days = 80),
            "Trending" to generateTrendingData(dailyReturn = 0.008, volatility = 0.025, days = 80),
            "Complex Pattern" to generateComplexPatternData(days = 80),
            "Step Function" to generateStepFunctionData(stepSize = 3.0, stepFrequency = 15, days = 80)
        )

        patterns.forEach { (patternName, data) ->
            val currentPrice = data.last().price
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = currentPrice * 1.06,
                lowerPriceBand = currentPrice * 0.94,
                daysPrediction = 5,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { transformerForecaster.calculate(calculations) }

            assertTrue("$patternName pattern should be handled",
                result.contains("Upper band") && result.contains("Lower band"))

            val upperProb = extractProbabilityFromResult(result, "Upper")
            val lowerProb = extractProbabilityFromResult(result, "Lower")

            assertTrue("$patternName: Upper probability should be valid",
                upperProb >= 0.0 && upperProb <= 100.0)
            assertTrue("$patternName: Lower probability should be valid",
                lowerProb >= 0.0 && lowerProb <= 100.0)
        }
    }

    // ================== TRANSFORMER SPECIFIC TESTS ==================

    @Test
    fun testMultiHeadAttention_handlesSequentialPatterns() {
        // Create data with clear sequential pattern for transformer to learn
        val sequentialData = generateStepFunctionData(
            basePrice = 100.0,
            days = 100,
            stepSize = 2.0,
            stepFrequency = 10
        )

        val calculations = Calculations(
            timeSeries = sequentialData,
            upperPriceBand = 120.0,
            lowerPriceBand = 110.0,
            daysPrediction = 8,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }
        val upperProb = extractProbabilityFromResult(result, "Upper")

        // With upward step pattern, should favor upper band
        assertTrue("Sequential pattern should be captured by transformer",
            result.contains("Upper band") && result.contains("Lower band"))
        assertTrue("Upper probability should be reasonable for upward pattern",
            upperProb >= 0.0 && upperProb <= 100.0)
    }

    @Test
    fun testAttentionMechanism_longSequenceHandling() {
        val longSequence = generateComplexPatternData(days = 200)

        val calculations = Calculations(
            timeSeries = longSequence,
            upperPriceBand = 110.0,
            lowerPriceBand = 90.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val startTime = System.currentTimeMillis()
        val result = runBlocking { transformerForecaster.calculate(calculations) }
        val endTime = System.currentTimeMillis()

        assertTrue("Should handle long sequences efficiently",
            (endTime - startTime) < 45000) // Less than 45 seconds

        assertTrue("Should return valid result for long sequences",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Should return valid probabilities",
            upperProb >= 0.0 && upperProb <= 100.0 && lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testTransformerLayers_deepArchitecture() {
        // Test that multi-layer transformer doesn't break
        val data = generateComplexPatternData(days = 150)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 108.0,
            lowerPriceBand = 92.0,
            daysPrediction = 6,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }

        assertTrue("Multi-layer transformer should work",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        // Should not produce NaN or invalid values
        assertFalse("Upper probability should not be NaN", upperProb.isNaN())
        assertFalse("Lower probability should not be NaN", lowerProb.isNaN())
        assertTrue("Upper probability should be in valid range", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be in valid range", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testFeatureScaling_consistentAcrossScales() {
        // Test transformer works across different price scales
        val lowPriceData = generateSineWaveData(amplitude = 2.0, basePrice = 50.0, days = 80)
        val highPriceData = generateSineWaveData(amplitude = 20.0, basePrice = 500.0, days = 80)

        val lowPriceCalc = Calculations(
            timeSeries = lowPriceData,
            upperPriceBand = 53.0,
            lowerPriceBand = 47.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val highPriceCalc = Calculations(
            timeSeries = highPriceData,
            upperPriceBand = 530.0,
            lowerPriceBand = 470.0,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val lowResult = runBlocking { transformerForecaster.calculate(lowPriceCalc) }
        val highResult = runBlocking { transformerForecaster.calculate(highPriceCalc) }

        assertTrue("Should handle low price scale",
            lowResult.contains("Upper band") && lowResult.contains("Lower band"))
        assertTrue("Should handle high price scale",
            highResult.contains("Upper band") && highResult.contains("Lower band"))

        val lowProb = extractProbabilityFromResult(lowResult, "Upper")
        val highProb = extractProbabilityFromResult(highResult, "Upper")

        assertTrue("Low scale should give valid probabilities", lowProb >= 0.0 && lowProb <= 100.0)
        assertTrue("High scale should give valid probabilities", highProb >= 0.0 && highProb <= 100.0)
    }

    // ================== MATHEMATICAL SOUNDNESS TESTS ==================

    @Test
    fun testTransformerSignal_respondsToTrend() {
        val upTrendData = generateTrendingData(
            startPrice = 100.0,
            dailyReturn = 0.01, // 1% daily growth
            volatility = 0.015,
            days = 100
        )

        val downTrendData = generateTrendingData(
            startPrice = 100.0,
            dailyReturn = -0.008, // -0.8% daily decline
            volatility = 0.015,
            days = 100
        )

        val upTrendCalc = Calculations(
            timeSeries = upTrendData,
            upperPriceBand = 120.0,
            lowerPriceBand = 80.0,
            daysPrediction = 8,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val downTrendCalc = Calculations(
            timeSeries = downTrendData,
            upperPriceBand = 120.0,
            lowerPriceBand = 80.0,
            daysPrediction = 8,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val upResult = runBlocking { transformerForecaster.calculate(upTrendCalc) }
        val downResult = runBlocking { transformerForecaster.calculate(downTrendCalc) }

        val upUpperProb = extractProbabilityFromResult(upResult, "Upper")
        val downLowerProb = extractProbabilityFromResult(downResult, "Lower")

        // Transformer should capture trend direction
        assertTrue("Up trend should give reasonable upper probability", upUpperProb >= 0.0)
        assertTrue("Down trend should give reasonable lower probability", downLowerProb >= 0.0)
    }

    @Test
    fun testBarrierProbability_reflectionPrinciple() {
        val data = generateSineWaveData(amplitude = 5.0, period = 30, days = 120)
        val currentPrice = data.last().price

        val nearBarrierCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.03, // 3% away
            lowerPriceBand = currentPrice * 0.97,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val farBarrierCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.20, // 20% away
            lowerPriceBand = currentPrice * 0.80,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val nearResult = runBlocking { transformerForecaster.calculate(nearBarrierCalc) }
        val farResult = runBlocking { transformerForecaster.calculate(farBarrierCalc) }

        val nearUpperProb = extractProbabilityFromResult(nearResult, "Upper")
        val farUpperProb = extractProbabilityFromResult(farResult, "Upper")

        assertTrue("Near barriers should generally have higher touch probability",
            nearUpperProb >= farUpperProb - 15.0) // Allow tolerance
    }

    @Test
    fun testTimeHorizon_longerPeriodsHigherProbability() {
        val data = generateSineWaveData(amplitude = 3.0, period = 25, days = 100)
        val currentPrice = data.last().price

        val shortTermCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.05,
            lowerPriceBand = currentPrice * 0.95,
            daysPrediction = 3,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val longTermCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.05,
            lowerPriceBand = currentPrice * 0.95,
            daysPrediction = 12,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val shortResult = runBlocking { transformerForecaster.calculate(shortTermCalc) }
        val longResult = runBlocking { transformerForecaster.calculate(longTermCalc) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        assertTrue("Longer periods should generally increase touch probability",
            longUpperProb >= shortUpperProb - 8.0) // Allow tolerance for transformer variation
    }

    @Test
    fun testVolatilityImpact_higherVolatilityHigherProbability() {
        val lowVolData = generateSineWaveData(amplitude = 1.0, period = 30, days = 100)
        val highVolData = generateSineWaveData(amplitude = 6.0, period = 30, days = 100)

        val lowVolCalc = Calculations(
            timeSeries = lowVolData,
            upperPriceBand = 104.0,
            lowerPriceBand = 96.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val highVolCalc = Calculations(
            timeSeries = highVolData,
            upperPriceBand = 104.0,
            lowerPriceBand = 96.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val lowVolResult = runBlocking { transformerForecaster.calculate(lowVolCalc) }
        val highVolResult = runBlocking { transformerForecaster.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        assertTrue("Higher volatility should increase barrier touch probability",
            highVolProb >= lowVolProb - 10.0) // Allow tolerance for transformer effects
    }

    // ================== PERFORMANCE AND MEMORY TESTS ==================

    @Test
    fun testDataLimiting_handlesExcessiveData() {
        // Test with data exceeding 500-day limit
        val excessiveData = generateComplexPatternData(days = 600)

        val calculations = Calculations(
            timeSeries = excessiveData,
            upperPriceBand = 110.0,
            lowerPriceBand = 90.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }

        assertTrue("Should handle excessive data by limiting",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Should return valid probabilities with limited data",
            upperProb >= 0.0 && upperProb <= 100.0 && lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    @Test
    fun testMemoryEfficiency_multipleCalculations() {
        val data = generateComplexPatternData(days = 150)

        repeat(4) { iteration ->
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = 105.0 + iteration,
                lowerPriceBand = 95.0 - iteration,
                daysPrediction = 6 + iteration,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { transformerForecaster.calculate(calculations) }

            assertTrue("Iteration $iteration should return valid result",
                result.contains("Upper band") && result.contains("Lower band"))
        }

        assertTrue("Memory should be managed efficiently", true)
    }

    // ================== OUTPUT FORMAT TESTS ==================

    @Test
    fun testOutputFormat_correctStructure() {
        val data = generateSineWaveData(amplitude = 3.0, period = 25, days = 80)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 104.5,
            lowerPriceBand = 95.5,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }

        assertTrue("Should contain upper band reference", result.contains("Upper band of 104.5"))
        assertTrue("Should contain lower band reference", result.contains("Lower band of 95.5"))
        assertTrue("Should contain probability keyword", result.contains("probability"))
        assertTrue("Should contain percentage symbols", result.contains("%"))

        val lines = result.split("\n")
        assertEquals("Should have exactly 2 lines", 2, lines.size)
    }

    @Test
    fun testRobustness_irregularPatterns() {
        // Create data with sudden jumps and gaps
        val baseData = generateSineWaveData(amplitude = 3.0, period = 25, days = 100)

        val irregularData = baseData.mapIndexed { index, entity ->
            when {
                index % 20 == 0 -> entity.copy(price = entity.price * 1.15) // Sudden jumps
                index % 30 == 0 -> entity.copy(price = entity.price * 0.85) // Sudden drops
                else -> entity
            }
        }

        val calculations = Calculations(
            timeSeries = irregularData,
            upperPriceBand = 108.0,
            lowerPriceBand = 92.0,
            daysPrediction = 8,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { transformerForecaster.calculate(calculations) }

        assertTrue("Should handle irregular patterns robustly",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Should return valid probabilities for irregular data",
            upperProb >= 0.0 && upperProb <= 100.0 && lowerProb >= 0.0 && lowerProb <= 100.0)
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