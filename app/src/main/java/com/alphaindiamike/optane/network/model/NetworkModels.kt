package com.alphaindiamike.optane.network.model

// Network response models (data classes for JSON parsing)
data class InstrumentChartResponse(
    val series: SeriesData
)

data class SeriesData(
    val history: HistoryData
)

data class HistoryData(
    val data: List<List<Double>>  // Array of [timestamp, price] pairs
)

// Network request model
data class HistoryRequest(
    val instrumentId: String,
    val marketId: Int = 1,
    val quotetype: String = "mid",
    val series: String = "history",
    val localeId: Int = 2
)

data class RawTimeSeriesData(
    val timestamp: Long,
    val price: Double
)