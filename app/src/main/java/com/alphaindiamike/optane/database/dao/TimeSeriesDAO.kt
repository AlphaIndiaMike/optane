package com.alphaindiamike.optane.database.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity

@Dao
interface TimeSeriesDAO {
    @Query("SELECT * FROM time_series_data WHERE asset_id = :assetId ORDER BY date ASC")
    fun getTimeSeriesForAsset(assetId: Long): List<TimeSeriesEntity>

    @Query("SELECT * FROM time_series_data WHERE asset_id = :assetId AND date >= :startDate AND date <= :endDate ORDER BY date ASC")
    fun getTimeSeriesForAssetInRange(assetId: Long, startDate: Long, endDate: Long): List<TimeSeriesEntity>

    @Query("SELECT * FROM time_series_data WHERE asset_id = :assetId ORDER BY date ASC LIMIT 1")
    fun getLatestPriceForAsset(assetId: Long): TimeSeriesEntity?

    @Query("SELECT * FROM time_series_data WHERE asset_id = :assetId AND date = :date LIMIT 1")
    fun getPriceForAssetAtDate(assetId: Long, date: Long): TimeSeriesEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertTimeSeriesData(data: TimeSeriesEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertTimeSeriesData(dataList: List<TimeSeriesEntity>): List<Long>

    @Delete
    fun deleteTimeSeriesData(data: TimeSeriesEntity): Int

    @Query("DELETE FROM time_series_data WHERE asset_id = :assetId")
    fun deleteAllTimeSeriesForAsset(assetId: Long): Int

    @Query("DELETE FROM time_series_data WHERE asset_id = :assetId AND date >= :startDate AND date <= :endDate")
    fun deleteTimeSeriesForAssetInRange(assetId: Long, startDate: Long, endDate: Long): Int

    @Query("SELECT COUNT(*) FROM time_series_data WHERE asset_id = :assetId")
    fun getTimeSeriesCountForAsset(assetId: Long): Int
}