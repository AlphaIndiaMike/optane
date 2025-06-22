package com.alphaindiamike.optane

import org.junit.runner.RunWith
import org.junit.runners.Suite

/**
 * Combined test suite for all forecasting algorithms
 * Run this to execute all algorithm tests together
 */
@RunWith(Suite::class)
@Suite.SuiteClasses(
    ProbabilisticForecasterTest::class,
    MonteCarloBasicTest::class,
    MonteCarloAdvancedTest::class,
    QuantileTransformerForecasterTest::class,
    TransformerForecasterTest::class,
    RegimeSwitchingForecasterTest::class,
    JumpDiffusionForecasterTest::class
)
class AlgorithmTestSuite {
    // This class remains empty, it's just a holder for the above annotations
}