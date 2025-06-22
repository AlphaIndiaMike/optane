package com.alphaindiamike.optane.database.entities


import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "main_asset_dict")
data class MainAssetEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "name")
    val name: String,

    @ColumnInfo(name = "exchange_id")
    val exchangeId: String,

    @ColumnInfo(name = "last_update")
    val lastUpdate: Long? // timestamp in milliseconds , nullable
)