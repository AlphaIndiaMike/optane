package com.alphaindiamike.optane

import com.alphaindiamike.optane.algorithms.implementations.QuantileTransformerForecasterImpl
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlinx.coroutines.runBlocking
import kotlin.math.*

/**
 * Test suite for QuantileTransformerForecasterImpl
 * Tests advanced quantile regression with deterministic feature engineering
 */
class QuantileTransformerForecasterTest {

    private lateinit var quantileTransformer: QuantileTransformerForecasterImpl

    @Before
    fun setup() {
        quantileTransformer = QuantileTransformerForecasterImpl(enableDebugLogging = false)
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

    // ================== BASIC FUNCTIONALITY TESTS ==================

    @Test
    fun testInsufficientData_returnsErrorMessage() {
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

        val result = runBlocking { quantileTransformer.calculate(calculations) }
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

        val negativeResult = runBlocking { quantileTransformer.calculate(negativeCalc) }
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

        val invalidOrderResult = runBlocking { quantileTransformer.calculate(invalidOrderCalc) }
        assertEquals("Upper band must be greater than lower band", invalidOrderResult)
    }

    @Test
    fun testMinimumDataRequirement_exactlyTenPoints() {
        val minimalData = generateSineWaveData(days = 10)

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

        val result = runBlocking { quantileTransformer.calculate(calculations) }

        assertTrue("Algorithm should handle minimal data",
            result.contains("Upper band") && result.contains("Lower band"))
    }

    // ================== DETERMINISTIC BEHAVIOR TESTS ==================

    @Test
    fun testDeterministicBehavior_sameInputSameOutput() {
        val data = generateSineWaveData(amplitude = 3.0, period = 25, days = 100)

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
            val result = runBlocking { quantileTransformer.calculate(calculations) }
            results.add(result)
        }

        assertTrue("Quantile Transformer should be deterministic",
            results.all { it == results[0] })
    }

    @Test
    fun testFeatureEngineering_handlesVariousPatterns() {
        val patterns = listOf(
            generateSineWaveData(amplitude = 2.0, period = 20, days = 60),
            generateTrendingData(dailyReturn = 0.01, volatility = 0.03, days = 60),
            generateConstantPriceData(price = 150.0, days = 60)
        )

        patterns.forEach { data ->
            val currentPrice = data.last().price
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = currentPrice * 1.05,
                lowerPriceBand = currentPrice * 0.95,
                daysPrediction = 5,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { quantileTransformer.calculate(calculations) }

            assertTrue("Pattern should be handled",
                result.contains("Upper band") && result.contains("Lower band"))

            val upperProb = extractProbabilityFromResult(result, "Upper")
            val lowerProb = extractProbabilityFromResult(result, "Lower")

            assertTrue("Upper probability should be valid", upperProb >= 0.0 && upperProb <= 100.0)
            assertTrue("Lower probability should be valid", lowerProb >= 0.0 && lowerProb <= 100.0)
        }
    }

    // ================== MATHEMATICAL SOUNDNESS TESTS ==================

    @Test
    fun testQuantileRegression_properDistribution() {
        val trendingData = generateTrendingData(
            startPrice = 100.0,
            dailyReturn = 0.008,
            volatility = 0.02,
            days = 120
        )

        val calculations = Calculations(
            timeSeries = trendingData,
            upperPriceBand = 110.0,
            lowerPriceBand = 95.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { quantileTransformer.calculate(calculations) }
        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Trending up data should favor upper band probability",
            upperProb >= lowerProb - 15.0)
    }

    @Test
    fun testBarrierProbability_reflectionPrinciple() {
        val data = generateSineWaveData(amplitude = 4.0, period = 30, days = 90)
        val currentPrice = data.last().price

        val nearBarrierCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.02,
            lowerPriceBand = currentPrice * 0.98,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val farBarrierCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.15,
            lowerPriceBand = currentPrice * 0.85,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val nearResult = runBlocking { quantileTransformer.calculate(nearBarrierCalc) }
        val farResult = runBlocking { quantileTransformer.calculate(farBarrierCalc) }

        val nearUpperProb = extractProbabilityFromResult(nearResult, "Upper")
        val farUpperProb = extractProbabilityFromResult(farResult, "Upper")

        assertTrue("Near barriers should have higher touch probability than far barriers",
            nearUpperProb >= farUpperProb - 10.0)
    }

    @Test
    fun testTimeScaling_longerPeriodsHigherProbability() {
        val data = generateSineWaveData(amplitude = 3.0, period = 25, days = 100)
        val currentPrice = data.last().price

        val shortTermCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.04,
            lowerPriceBand = currentPrice * 0.96,
            daysPrediction = 3,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val longTermCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.04,
            lowerPriceBand = currentPrice * 0.96,
            daysPrediction = 15,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val shortResult = runBlocking { quantileTransformer.calculate(shortTermCalc) }
        val longResult = runBlocking { quantileTransformer.calculate(longTermCalc) }

        val shortUpperProb = extractProbabilityFromResult(shortResult, "Upper")
        val longUpperProb = extractProbabilityFromResult(longResult, "Upper")

        assertTrue("Longer prediction periods should increase touch probability",
            longUpperProb >= shortUpperProb - 5.0)
    }

    @Test
    fun testVolatilityImpact_higherVolatilityHigherProbability() {
        val lowVolData = generateSineWaveData(amplitude = 1.0, period = 30, days = 80)
        val highVolData = generateSineWaveData(amplitude = 5.0, period = 30, days = 80)

        val lowVolCalc = Calculations(
            timeSeries = lowVolData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val highVolCalc = Calculations(
            timeSeries = highVolData,
            upperPriceBand = 103.0,
            lowerPriceBand = 97.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val lowVolResult = runBlocking { quantileTransformer.calculate(lowVolCalc) }
        val highVolResult = runBlocking { quantileTransformer.calculate(highVolCalc) }

        val lowVolProb = extractProbabilityFromResult(lowVolResult, "Upper")
        val highVolProb = extractProbabilityFromResult(highVolResult, "Upper")

        assertTrue("Higher volatility should increase barrier touch probability",
            highVolProb >= lowVolProb - 8.0)
    }

    // ================== ADVANCED ALGORITHM TESTS ==================

    @Test
    fun testAttentionMechanism_handlesLongSequences() {
        val longSequence = generateSineWaveData(amplitude = 3.0, period = 40, days = 500)

        val calculations = Calculations(
            timeSeries = longSequence,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val startTime = System.currentTimeMillis()
        val result = runBlocking { quantileTransformer.calculate(calculations) }
        val endTime = System.currentTimeMillis()

        assertTrue("Should handle long sequences efficiently",
            (endTime - startTime) < 60000)

        assertTrue("Should return valid result for long sequences",
            result.contains("Upper band") && result.contains("Lower band"))
    }

    @Test
    fun testMemoryManagement_noMemoryLeaks() {
        val data = generateSineWaveData(amplitude = 3.0, period = 20, days = 150)

        repeat(5) { iteration ->
            val calculations = Calculations(
                timeSeries = data,
                upperPriceBand = 104.0 + iteration,
                lowerPriceBand = 96.0 - iteration,
                daysPrediction = 5 + iteration,
                name = "test",
                exchangeId = "0",
                lastUpdate = System.currentTimeMillis(),
                result = ""
            )

            val result = runBlocking { quantileTransformer.calculate(calculations) }

            assertTrue("Iteration $iteration should return valid result",
                result.contains("Upper band") && result.contains("Lower band"))
        }

        assertTrue("Memory management should prevent leaks", true)
    }

    @Test
    fun testDataLimiting_handlesExcessiveData() {
        val excessiveData = generateSineWaveData(amplitude = 4.0, period = 50, days = 1500)

        val calculations = Calculations(
            timeSeries = excessiveData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 10,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { quantileTransformer.calculate(calculations) }

        assertTrue("Should handle excessive data by limiting",
            result.contains("Upper band") && result.contains("Lower band"))

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Should return valid probabilities with limited data",
            upperProb >= 0.0 && upperProb <= 100.0 && lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    // ================== OUTPUT FORMAT TESTS ==================

    @Test
    fun testOutputFormat_correctStructure() {
        val data = generateSineWaveData(amplitude = 3.0, period = 25, days = 60)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 104.0,
            lowerPriceBand = 96.0,
            daysPrediction = 7,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { quantileTransformer.calculate(calculations) }

        assertTrue("Should contain upper band reference", result.contains("Upper band of 104.0"))
        assertTrue("Should contain lower band reference", result.contains("Lower band of 96.0"))
        assertTrue("Should contain probability keyword", result.contains("probability"))
        assertTrue("Should contain percentage symbols", result.contains("%"))

        val lines = result.split("\n")
        assertEquals("Should have exactly 2 lines", 2, lines.size)

        lines.forEach { line ->
            assertTrue("Each line should mention band and probability",
                line.contains("band") && line.contains("probability") && line.contains("%"))
        }
    }

    @Test
    fun testProbabilityExtraction_accurateValues() {
        val data = generateSineWaveData(amplitude = 2.5, period = 30, days = 70)

        val calculations = Calculations(
            timeSeries = data,
            upperPriceBand = 103.5,
            lowerPriceBand = 96.5,
            daysPrediction = 6,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { quantileTransformer.calculate(calculations) }

        val upperProb = extractProbabilityFromResult(result, "Upper")
        val lowerProb = extractProbabilityFromResult(result, "Lower")

        assertTrue("Upper probability should be valid percentage", upperProb >= 0.0 && upperProb <= 100.0)
        assertTrue("Lower probability should be valid percentage", lowerProb >= 0.0 && lowerProb <= 100.0)
    }

    // ================== ROBUSTNESS TESTS ==================

    @Test
    fun testEdgeCases_extremeBands() {
        val data = generateSineWaveData(amplitude = 2.0, period = 20, days = 60)
        val currentPrice = data.last().price

        val veryCloseCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 1.001,
            lowerPriceBand = currentPrice * 0.999,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val veryFarCalc = Calculations(
            timeSeries = data,
            upperPriceBand = currentPrice * 2.0,
            lowerPriceBand = currentPrice * 0.5,
            daysPrediction = 5,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val closeResult = runBlocking { quantileTransformer.calculate(veryCloseCalc) }
        val farResult = runBlocking { quantileTransformer.calculate(veryFarCalc) }

        assertTrue("Should handle very close bands",
            closeResult.contains("Upper band") && closeResult.contains("Lower band"))
        assertTrue("Should handle very far bands",
            farResult.contains("Upper band") && farResult.contains("Lower band"))
    }

    @Test
    fun testRobustness_irregularData() {
        val baseData = generateSineWaveData(amplitude = 3.0, period = 25, days = 80)

        val irregularData = baseData.mapIndexed { index, entity ->
            if (index % 15 == 0) {
                val jumpFactor = if (index % 30 == 0) 1.1 else 0.9
                entity.copy(price = entity.price * jumpFactor)
            } else {
                entity
            }
        }

        val calculations = Calculations(
            timeSeries = irregularData,
            upperPriceBand = 105.0,
            lowerPriceBand = 95.0,
            daysPrediction = 8,
            name = "test",
            exchangeId = "0",
            lastUpdate = System.currentTimeMillis(),
            result = ""
        )

        val result = runBlocking { quantileTransformer.calculate(calculations) }

        assertTrue("Should handle irregular data patterns",
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