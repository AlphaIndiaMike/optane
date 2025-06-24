package com.alphaindiamike.optane.network

import com.alphaindiamike.optane.network.model.RawTimeSeriesData

interface NetworkCapRepository {
    suspend fun fetchNetworkData(instrumentId: String): List<RawTimeSeriesData>

}