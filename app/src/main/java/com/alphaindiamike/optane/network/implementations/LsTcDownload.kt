package com.alphaindiamike.optane.network.implementations

import com.alphaindiamike.optane.network.NetworkCapRepository
import com.alphaindiamike.optane.network.model.NetworkException
import com.alphaindiamike.optane.network.model.HistoryRequest
import com.alphaindiamike.optane.network.model.InstrumentChartResponse
import com.alphaindiamike.optane.network.model.RawTimeSeriesData
import com.google.gson.Gson
import java.net.URL
import java.net.HttpURLConnection
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext


class LsTcDownload : NetworkCapRepository{
    private val gson = Gson()
    private val baseUrl = "https://www.ls-tc.de/_rpc/json/instrument/chart/dataForInstrument"

    // Headers to show we are legit browser
    private val headers = mapOf(
        "User-Agent" to "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept" to "application/json, text/plain, */*",
        "Accept-Language" to "en-US,en;q=0.9"
    )

    override suspend fun fetchNetworkData(instrumentId: String): List<RawTimeSeriesData> {
        return withContext(
            // Coroutine Context
            Dispatchers.IO,
            // Suspend function block
            block = {
                try {
                    val response = makeNetworkRequest(instrumentId)
                    parseResponseToTimeSeriesData(response)
                } catch (e: Exception) {
                    throw NetworkException("Failed to fetch history data for instrument: $instrumentId", e)
                }
            }
        )
    }

    private fun makeNetworkRequest(instrumentId: String): InstrumentChartResponse {
        val urlString = buildUrl(instrumentId)
        val url = URL(urlString)
        val connection = url.openConnection() as HttpURLConnection

        try {
            // Set headers
            headers.forEach { (key, value) ->
                connection.setRequestProperty(key, value)
            }

            connection.requestMethod = "GET"
            connection.connectTimeout = 10000 // 10 seconds
            connection.readTimeout = 30000    // 30 seconds

            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw NetworkException("HTTP Error: $responseCode")
            }

            val reader = BufferedReader(InputStreamReader(connection.inputStream))
            val response = reader.use { it.readText() }

            return gson.fromJson(response, InstrumentChartResponse::class.java)

        } finally {
            connection.disconnect()
        }
    }

    private fun buildUrl(instrumentId: String): String {
        val request = HistoryRequest(instrumentId = instrumentId)
        return "$baseUrl?instrumentId=${request.instrumentId}" +
                "&marketId=${request.marketId}" +
                "&quotetype=${request.quotetype}" +
                "&series=${request.series}" +
                "&localeId=${request.localeId}"
    }

    private fun parseResponseToTimeSeriesData(
        response: InstrumentChartResponse
    ): List<RawTimeSeriesData> {

        return response.series.history.data.map { dataPoint ->
            // dataPoint is [timestamp, price] array
            val timestamp = dataPoint[0].toLong() * 1000000 // Convert to microseconds like Python
            val price = dataPoint[1]

            RawTimeSeriesData(
                timestamp = timestamp,
                price = price
            )
        }.sortedBy(selector = { it.timestamp })
    }
}