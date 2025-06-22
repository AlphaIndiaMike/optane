package com.alphaindiamike.optane.network

import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.network.model.RawTimeSeriesData

interface NetworkCapRepository {
    suspend fun downloadLsTcSingleAssetData(instrumentId: String): List<RawTimeSeriesData>

}