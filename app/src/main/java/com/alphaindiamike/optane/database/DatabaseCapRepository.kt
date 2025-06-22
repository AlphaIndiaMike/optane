package com.alphaindiamike.optane.database

import com.alphaindiamike.optane.database.entities.MainAssetEntity
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity

class DatabaseCapRepository(private val database: DatabaseItr) {
    private val mainAssetDao = database.mainAssetDao()
    private val timeSeriesDao = database.timeSeriesDao()

    // Main Asset operations
    fun getAllAssets(): List<MainAssetEntity> {
        return mainAssetDao.getAllAssets()
    }

    fun getAssetById(assetId: Long): MainAssetEntity? {
        return mainAssetDao.getAssetById(assetId)
    }

    fun insertAsset(asset: MainAssetEntity): Long {
        return mainAssetDao.insertAsset(asset)
    }

    fun insertAssetSkipExisting(asset: MainAssetEntity): Long {
        val existing = mainAssetDao.getAssetByExchangeId(asset.exchangeId)
        return existing?.id ?: mainAssetDao.insertAsset(asset)
    }

    fun deleteAsset(assetId: Long): Int {
        val asset: MainAssetEntity? = mainAssetDao.getAssetById(assetId);
        if (asset != null) {
            return mainAssetDao.deleteAsset(asset)
        }
        return 1;
    }

    fun updateAssetLastUpdate(assetId: Long, timestamp: Long): Int {
        return mainAssetDao.updateLastUpdate(assetId, timestamp)
    }

    // Time Series operations
    fun getTimeSeriesForAsset(assetId: Long): List<TimeSeriesEntity> {
        return timeSeriesDao.getTimeSeriesForAsset(assetId)
    }

    // Time Series operations
    fun getTimeSeriesForAssetLastDataPoint(assetId: Long): TimeSeriesEntity {
        return timeSeriesDao.getTimeSeriesForAsset(assetId).last()
    }

    fun insertTimeSeriesData(dataList: List<TimeSeriesEntity>): List<Long> {
        return timeSeriesDao.insertTimeSeriesData(dataList)
    }

    fun getLatestPriceForAsset(assetId: Long): TimeSeriesEntity? {
        return timeSeriesDao.getLatestPriceForAsset(assetId)
    }

    fun deleteAllTimeSeriesForAsset(assetId: Long): Int {
        return timeSeriesDao.deleteAllTimeSeriesForAsset(assetId)
    }
}