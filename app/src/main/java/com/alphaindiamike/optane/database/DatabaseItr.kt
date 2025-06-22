package com.alphaindiamike.optane.database

import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import android.content.Context
import com.alphaindiamike.optane.database.entities.MainAssetEntity
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.database.dao.MainAssetDAO
import com.alphaindiamike.optane.database.dao.TimeSeriesDAO

@Database(
    entities = [MainAssetEntity::class, TimeSeriesEntity::class],
    version = 1,
    exportSchema = true
)
abstract class DatabaseItr : RoomDatabase() {

    abstract fun mainAssetDao(): MainAssetDAO
    abstract fun timeSeriesDao(): TimeSeriesDAO

    companion object {
        @Volatile
        private var INSTANCE: DatabaseItr? = null

        private const val DATABASE_NAME = "optane_asset_database"

        fun getDatabase(context: Context): DatabaseItr {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    DatabaseItr::class.java,
                    DATABASE_NAME
                ).build();

                INSTANCE = instance
                instance
            }
        }
    }
}