package com.alphaindiamike.optane.database.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.alphaindiamike.optane.database.entities.MainAssetEntity

@Dao
interface MainAssetDAO {
    @Query("SELECT * FROM main_asset_dict")
    fun getAllAssets(): List<MainAssetEntity>

    @Query("SELECT * FROM main_asset_dict WHERE id = :assetId")
    fun getAssetById(assetId: Long): MainAssetEntity?

    @Query("SELECT * FROM main_asset_dict WHERE exchange_id = :exchangeId")
    fun getAssetsByExchange(exchangeId: String): List<MainAssetEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertAsset(asset: MainAssetEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertAssets(assets: List<MainAssetEntity>): List<Long>

    @Update
    fun updateAsset(asset: MainAssetEntity): Int

    @Delete
    fun deleteAsset(asset: MainAssetEntity): Int

    @Query("UPDATE main_asset_dict SET last_update = :timestamp WHERE id = :assetId")
    fun updateLastUpdate(assetId: Long, timestamp: Long): Int

    @Query("SELECT * from main_asset_dict WHERE exchange_id = :exchangeId LIMIT 1")
    fun getAssetByExchangeId(exchangeId: String): MainAssetEntity?

}