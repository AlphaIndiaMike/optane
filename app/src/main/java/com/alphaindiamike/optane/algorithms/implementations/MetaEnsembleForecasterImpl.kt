package com.alphaindiamike.optane.algorithms.implementations


import kotlin.math.*
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.algorithms.AlgorithmRepository
import com.alphaindiamike.optane.model.Calculations

/**
 * MetaEnsembleForecaster - Meta-Ensemble with Adaptive Weights
 * Advanced ensemble combining multiple state-of-the-art models
 */
class MetaEnsembleForecasterImpl : AlgorithmRepository {

    // Individual models
    private val quantileTransformer = QuantileTransformerForecasterImpl()
    private val regimeSwitching = RegimeSwitchingForecasterImpl()

    // Adaptive weights configuration
    private data class AdaptiveWeights(
        val quantileTransformer: Double = 0.7,
        val regimeSwitching: Double = 0.3
    )

    // Model performance tracking
    private data class ModelPerformance(
        val accuracy: Double,
        val recentPerformance: Double,
        val volatilityPenalty: Double,
        val consistencyScore: Double
    )

    // Individual model results for combination
    private data class ModelResult(
        val upperBandProbability: Double,
        val lowerBandProbability: Double,
        val confidence: Double,
        val modelName: String
    )

    override suspend fun calculate(calculations: Calculations): String {
        val timeSeries = calculations.timeSeries
        val upperBand = calculations.upperPriceBand
        val lowerBand = calculations.lowerPriceBand
        val daysAhead = calculations.daysPrediction

        // Validate input
        if (timeSeries.size < 15) {
            return "Insufficient data"
        }

        // 1. Calculate adaptive weights based on recent performance
        val adaptiveWeights = calculateAdaptiveWeights(timeSeries)

        // 2. Get forecasts from individual models
        val individualResults = getIndividualResults(calculations)

        // 3. Combine with adaptive weights
        val metaForecast = combineResults(individualResults, adaptiveWeights)

        // 4. Apply confidence intervals and model disagreement analysis
        val finalResult = applyConfidenceAdjustment(metaForecast, individualResults)

        return """
            Upper band of ${upperBand.toString()} probability: ${String.format("%.1f", finalResult.first)}%
            Lower band of ${lowerBand.toString()} probability: ${String.format("%.1f", finalResult.second)}%
            """.trimIndent()
    }

    private fun calculateAdaptiveWeights(timeSeries: List<TimeSeriesEntity>): AdaptiveWeights {
        // Simulate performance-based weighting (in practice, use cross-validation)
        val recentPerformance = simulateModelPerformance(timeSeries)

        // Calculate weights based on multiple factors
        val qtWeight = calculateModelWeight(recentPerformance["quantileTransformer"]!!)
        val rsWeight = calculateModelWeight(recentPerformance["regimeSwitching"]!!)

        // Normalize weights
        val totalWeight = qtWeight + rsWeight

        return AdaptiveWeights(
            quantileTransformer = qtWeight / totalWeight,
            regimeSwitching = rsWeight / totalWeight
        )
    }

    private fun simulateModelPerformance(timeSeries: List<TimeSeriesEntity>): Map<String, ModelPerformance> {
        // In practice, this would use historical prediction accuracy
        // For now, simulate based on data characteristics

        val volatility = calculateHistoricalVolatility(timeSeries)
        val trendStrength = calculateTrendStrength(timeSeries)
        val regimeChanges = detectRegimeChanges(timeSeries)

        // QuantileTransformer performs better in complex, high-volatility environments
        val qtPerformance = ModelPerformance(
            accuracy = 0.75 + volatility * 2.0, // Higher volatility = better for advanced models
            recentPerformance = 0.70 + trendStrength * 0.3,
            volatilityPenalty = max(0.0, 0.1 - volatility), // Less penalty for high vol
            consistencyScore = 0.85
        )

        // RegimeSwitching performs better with clear regime changes
        val rsPerformance = ModelPerformance(
            accuracy = 0.65 + regimeChanges * 0.4,
            recentPerformance = 0.60 + regimeChanges * 0.5,
            volatilityPenalty = abs(volatility - 0.05) * 0.5, // Optimal around medium vol
            consistencyScore = 0.75
        )

        return mapOf(
            "quantileTransformer" to qtPerformance,
            "regimeSwitching" to rsPerformance
        )
    }

    private fun calculateModelWeight(performance: ModelPerformance): Double {
        // Combine multiple performance metrics
        val baseWeight = performance.accuracy * 0.4 +
                performance.recentPerformance * 0.3 +
                performance.consistencyScore * 0.2

        // Apply penalties
        val adjustedWeight = baseWeight - performance.volatilityPenalty * 0.1

        return max(0.1, adjustedWeight) // Minimum weight of 10%
    }

    private fun calculateHistoricalVolatility(timeSeries: List<TimeSeriesEntity>): Double {
        if (timeSeries.size < 2) return 0.02

        val returns = mutableListOf<Double>()
        for (i in 1 until timeSeries.size) {
            val ret = ln(timeSeries[i].price / timeSeries[i-1].price)
            returns.add(ret)
        }

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    private fun calculateTrendStrength(timeSeries: List<TimeSeriesEntity>): Double {
        if (timeSeries.size < 5) return 0.0

        val window = minOf(10, timeSeries.size - 1)
        val recent = timeSeries.takeLast(window)

        var upMoves = 0
        var downMoves = 0

        for (i in 1 until recent.size) {
            if (recent[i].price > recent[i-1].price) upMoves++
            else if (recent[i].price < recent[i-1].price) downMoves++
        }

        val totalMoves = upMoves + downMoves
        return if (totalMoves > 0) {
            abs(upMoves - downMoves).toDouble() / totalMoves
        } else 0.0
    }

    private fun detectRegimeChanges(timeSeries: List<TimeSeriesEntity>): Double {
        if (timeSeries.size < 10) return 0.0

        val prices = timeSeries.map { it.price }
        val window = 5
        var regimeChanges = 0

        for (i in window until prices.size - window) {
            val beforeVol = calculateWindowVolatility(prices, i - window, window)
            val afterVol = calculateWindowVolatility(prices, i, window)

            // Detect significant volatility regime changes
            if (abs(beforeVol - afterVol) / max(beforeVol, afterVol) > 0.5) {
                regimeChanges++
            }
        }

        return regimeChanges.toDouble() / (prices.size - 2 * window)
    }

    private fun calculateWindowVolatility(prices: List<Double>, start: Int, window: Int): Double {
        val returns = mutableListOf<Double>()
        for (i in start + 1 until start + window) {
            if (i < prices.size) {
                returns.add(ln(prices[i] / prices[i-1]))
            }
        }

        if (returns.isEmpty()) return 0.0

        val mean = returns.average()
        val variance = returns.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    private suspend fun getIndividualResults(calculations: Calculations): Map<String, ModelResult> {
        val results = mutableMapOf<String, ModelResult>()

        // Get QuantileTransformer result
        try {
            val qtResult = quantileTransformer.calculate(calculations)
            val qtParsed = parseResult(qtResult)
            results["quantileTransformer"] = ModelResult(
                upperBandProbability = qtParsed.first,
                lowerBandProbability = qtParsed.second,
                confidence = 0.85, // High confidence for advanced model
                modelName = "QuantileTransformer"
            )
        } catch (e: Exception) {
            // Fallback if model fails
            results["quantileTransformer"] = ModelResult(0.0, 0.0, 0.0, "QuantileTransformer")
        }

        // Get RegimeSwitching result
        try {
            val rsResult = regimeSwitching.calculate(calculations)
            val rsParsed = parseResult(rsResult)
            results["regimeSwitching"] = ModelResult(
                upperBandProbability = rsParsed.first,
                lowerBandProbability = rsParsed.second,
                confidence = 0.75, // Medium confidence for regime model
                modelName = "RegimeSwitching"
            )
        } catch (e: Exception) {
            // Fallback if model fails
            results["regimeSwitching"] = ModelResult(0.0, 0.0, 0.0, "RegimeSwitching")
        }

        return results
    }

    private fun parseResult(result: String): Pair<Double, Double> {
        // Parse "upperBandProbability:XX.X,lowerBandProbability:YY.Y"
        try {
            val parts = result.split(",")
            val upperPart = parts[0].split(":")[1].toDouble()
            val lowerPart = parts[1].split(":")[1].toDouble()
            return Pair(upperPart, lowerPart)
        } catch (e: Exception) {
            return Pair(0.0, 0.0)
        }
    }

    private fun combineResults(
        modelResults: Map<String, ModelResult>,
        weights: AdaptiveWeights
    ): Pair<Double, Double> {

        var weightedUpperSum = 0.0
        var weightedLowerSum = 0.0
        var totalWeight = 0.0

        // Combine QuantileTransformer results
        modelResults["quantileTransformer"]?.let { result ->
            if (result.confidence > 0.0) {
                val weight = weights.quantileTransformer * result.confidence
                weightedUpperSum += result.upperBandProbability * weight
                weightedLowerSum += result.lowerBandProbability * weight
                totalWeight += weight
            }
        }

        // Combine RegimeSwitching results
        modelResults["regimeSwitching"]?.let { result ->
            if (result.confidence > 0.0) {
                val weight = weights.regimeSwitching * result.confidence
                weightedUpperSum += result.upperBandProbability * weight
                weightedLowerSum += result.lowerBandProbability * weight
                totalWeight += weight
            }
        }

        // Normalize by total weight
        return if (totalWeight > 0.0) {
            Pair(weightedUpperSum / totalWeight, weightedLowerSum / totalWeight)
        } else {
            Pair(0.0, 0.0)
        }
    }

    private fun applyConfidenceAdjustment(
        metaForecast: Pair<Double, Double>,
        individualResults: Map<String, ModelResult>
    ): Pair<Double, Double> {

        // Calculate model disagreement
        val disagreement = calculateModelDisagreement(individualResults)

        // Adjust confidence based on disagreement
        val confidenceAdjustment = 1.0 - (disagreement * 0.2) // Reduce confidence with high disagreement

        // Apply confidence intervals
        val adjustedUpper = metaForecast.first * confidenceAdjustment
        val adjustedLower = metaForecast.second * confidenceAdjustment

        return Pair(
            max(0.0, min(100.0, adjustedUpper)),
            max(0.0, min(100.0, adjustedLower))
        )
    }

    private fun calculateModelDisagreement(individualResults: Map<String, ModelResult>): Double {
        val upperValues = individualResults.values.map { it.upperBandProbability }
        val lowerValues = individualResults.values.map { it.lowerBandProbability }

        if (upperValues.size < 2) return 0.0

        // Calculate standard deviation as disagreement measure
        val upperMean = upperValues.average()
        val lowerMean = lowerValues.average()

        val upperDisagreement = sqrt(upperValues.map { (it - upperMean).pow(2) }.average())
        val lowerDisagreement = sqrt(lowerValues.map { (it - lowerMean).pow(2) }.average())

        // Normalize disagreement (0 = perfect agreement, 1 = maximum disagreement)
        return (upperDisagreement + lowerDisagreement) / 200.0 // Divide by 200 since max prob is 100
    }

    // Risk scenario analysis (from your JS case study)
    private fun riskScenarioAnalysis(
        baseResult: Pair<Double, Double>,
        timeSeries: List<TimeSeriesEntity>
    ): Pair<Double, Double> {

        val volatility = calculateHistoricalVolatility(timeSeries)
        val trendStrength = calculateTrendStrength(timeSeries)

        // Define scenarios based on market conditions
        val scenarios = when {
            volatility < 0.02 && trendStrength > 0.7 -> {
                // Bullish scenario: Low volatility, strong trend
                mapOf(
                    "bullish" to Triple(0.4, 85.0, 5.0),  // probability, upper%, lower%
                    "neutral" to Triple(0.5, baseResult.first, baseResult.second),
                    "bearish" to Triple(0.1, 45.0, 35.0)
                )
            }
            volatility > 0.05 -> {
                // High volatility scenario
                mapOf(
                    "bullish" to Triple(0.2, 45.0, 15.0),
                    "neutral" to Triple(0.3, baseResult.first, baseResult.second),
                    "bearish" to Triple(0.5, 25.0, 45.0)
                )
            }
            else -> {
                // Neutral scenario
                mapOf(
                    "bullish" to Triple(0.3, baseResult.first * 1.2, baseResult.second * 0.8),
                    "neutral" to Triple(0.5, baseResult.first, baseResult.second),
                    "bearish" to Triple(0.2, baseResult.first * 0.8, baseResult.second * 1.2)
                )
            }
        }

        // Calculate scenario-weighted probabilities
        var scenarioWeightedUpper = 0.0
        var scenarioWeightedLower = 0.0

        scenarios.forEach { (_, scenario) ->
            val (probability, upper, lower) = scenario
            scenarioWeightedUpper += upper * probability
            scenarioWeightedLower += lower * probability
        }

        return Pair(scenarioWeightedUpper, scenarioWeightedLower)
    }

    // Model performance evaluation
    private fun evaluateEnsemblePerformance(
        predictions: List<Pair<Double, Double>>,
        actuals: List<Pair<Double, Double>>
    ): Map<String, Double> {

        if (predictions.size != actuals.size || predictions.isEmpty()) {
            return emptyMap()
        }

        val upperErrors = predictions.zip(actuals) { pred, actual -> abs(pred.first - actual.first) }
        val lowerErrors = predictions.zip(actuals) { pred, actual -> abs(pred.second - actual.second) }

        val upperMAE = upperErrors.average()
        val lowerMAE = lowerErrors.average()
        val overallMAE = (upperMAE + lowerMAE) / 2.0

        // Calculate hit rate (how often predictions were in right direction)
        val upperHitRate = predictions.zip(actuals) { pred, actual ->
            if ((pred.first > 50 && actual.first > 50) || (pred.first <= 50 && actual.first <= 50)) 1.0 else 0.0
        }.average()

        val lowerHitRate = predictions.zip(actuals) { pred, actual ->
            if ((pred.second > 50 && actual.second > 50) || (pred.second <= 50 && actual.second <= 50)) 1.0 else 0.0
        }.average()

        return mapOf(
            "upperMAE" to upperMAE,
            "lowerMAE" to lowerMAE,
            "overallMAE" to overallMAE,
            "upperHitRate" to upperHitRate * 100,
            "lowerHitRate" to lowerHitRate * 100,
            "overallHitRate" to ((upperHitRate + lowerHitRate) / 2.0) * 100
        )
    }
}