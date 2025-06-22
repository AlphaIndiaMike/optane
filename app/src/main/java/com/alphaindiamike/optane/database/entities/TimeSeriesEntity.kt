package com.alphaindiamike.optane.database.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "time_series_data",
    foreignKeys = [
        ForeignKey(
            entity = MainAssetEntity::class,
            parentColumns = ["id"],
            childColumns = ["asset_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["asset_id"]),
        Index(value = ["asset_id", "date"]),
        Index(value = ["date"])
    ]
)
data class TimeSeriesEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    @ColumnInfo(name = "asset_id")
    val assetId: Long,
    @ColumnInfo(name = "date")
    val date: Long, // timestamp in nano-seconds
    @ColumnInfo(name = "price")
    val price: Double
)
