package com.alphaindiamike.optane.services

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import com.alphaindiamike.optane.database.DatabaseCapRepository
import com.alphaindiamike.optane.database.DatabaseItr
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Notification
import android.app.PendingIntent
import android.util.Log
import androidx.core.app.NotificationCompat
import com.alphaindiamike.optane.MainActivity
import com.alphaindiamike.optane.database.entities.MainAssetEntity
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.network.NetworkCapRepository
import com.alphaindiamike.optane.network.implementations.LsTcDownload
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class UpdateAllSymbolsService : Service() {
    private val binder = UpdateBinder()
    private var isUpdating = false
    private lateinit var repository: DatabaseCapRepository

    companion object {
        const val ACTION_START_UPDATE = "START_UPDATE"
        const val ACTION_STOP_UPDATE = "STOP_UPDATE"
        const val NOTIFICATION_ID = 1001
        const val CHANNEL_ID = "update_channel"

        const val EXTRA_PROGRESS = "progress"
        const val EXTRA_SYMBOL_NAME = "symbol_name"
        const val EXTRA_TOTAL = "total"
        const val EXTRA_CURRENT = "current"

        const val BROADCAST_UPDATE_PROGRESS = "com.alphaindiamike.optane.UPDATE_PROGRESS"
        const val BROADCAST_UPDATE_COMPLETE = "com.alphaindiamike.optane.UPDATE_COMPLETE"
        const val BROADCAST_UPDATE_ERROR = "com.alphaindiamike.optane.UPDATE_ERROR"
        const val BROADCAST_UPDATE_CANCELLED = "com.alphaindiamike.optane.UPDATE_CANCELLED"
    }

    inner class UpdateBinder : Binder() {
        fun getService(): UpdateAllSymbolsService = this@UpdateAllSymbolsService
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    private fun setupDatabase() {
        val database = DatabaseItr.getDatabase(this)
        repository = DatabaseCapRepository(database)
    }


    override fun onCreate() {
        super.onCreate()
        setupDatabase()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_UPDATE -> {
                if (!isUpdating) {
                    startUpdateProcess()
                }
            }
            ACTION_STOP_UPDATE -> {
                stopUpdateProcess()
            }
        }
        return START_STICKY
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Symbol Updates",
            NotificationManager.IMPORTANCE_LOW
        )
        channel.description = "Updates symbols from network"

        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }

    private fun startUpdateProcess() {
        isUpdating = true

        val notification = createNotification("Starting update...", 0, 0)
        startForeground(NOTIFICATION_ID, notification)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val symbolsList = withContext(Dispatchers.IO) {
                    repository.getAllAssets()
                }
                updateSymbolsWithProgress(symbolsList)
            } catch (e: Exception) {
                broadcastError("Failed to load symbols: ${e.message}")
                stopSelf()
            }
        }
    }

    private suspend fun updateSymbolsWithProgress(symbolsList: List<MainAssetEntity>) {
        val totalSymbols = symbolsList.size
        var successCount = 0
        var failureCount = 0
        val failedSymbols = mutableListOf<String>()

        for (index in symbolsList.indices) {
            if (!isUpdating) {
                broadcastCancelled(successCount, failureCount, index)
                return
            }

            val symbol = symbolsList[index]

            // Update notification and broadcast progress
            val progress = ((index + 1) * 100) / totalSymbols
            updateNotification("Updating ${symbol.name}", progress, index + 1, totalSymbols)
            broadcastProgress(progress, symbol.name, index + 1, totalSymbols)

            try {
                // Random delay between 1-5 seconds
                val delay = (1000..5000).random()
                delay(delay.toLong())

                // Perform your network update here
                val success = updateSymbolFromNetwork(symbol)
                if (success) {
                    successCount++
                } else {
                    failureCount++
                    failedSymbols.add(symbol.name)
                }

            } catch (e: Exception) {
                Log.e("UpdateService", "Failed to update ${symbol.name}: ${e.message}")
                failureCount++
                failedSymbols.add(symbol.name)
            }
        }

        if (isUpdating) {
            broadcastComplete(successCount, failureCount, failedSymbols)
        }
        stopSelf()
    }

    private suspend fun updateSymbolFromNetwork(symbol: MainAssetEntity): Boolean {
        val symbolId = symbol.exchangeId.toString();
        val fkId = symbol.id;
        return try {
                // Step 1: Download raw data from network
                val networkRepository: NetworkCapRepository = LsTcDownload();
                val rawNetworkData = withContext(
                    // Coroutine Context
                    Dispatchers.IO,
                    // Suspend function block
                    block = {
                        networkRepository.downloadLsTcSingleAssetData(symbolId)
                    }
                )

                // Step 2: Convert raw data to TimeSeriesEntity
                val timeSeriesEntities = rawNetworkData.map { rawData ->
                    TimeSeriesEntity(
                        assetId = fkId,  // Use the foreign key available in Activity
                        date = rawData.timestamp,
                        price = rawData.price
                    )
                }

                // Step 3: Clear old data and insert new data using existing repository
                CoroutineScope(Dispatchers.IO).launch (
                    // Coroutine Context
                    Dispatchers.IO,
                    // Suspend function block
                    block = {
                        repository.deleteAllTimeSeriesForAsset(fkId)
                        repository.insertTimeSeriesData(timeSeriesEntities)
                        repository.updateAssetLastUpdate(fkId, System.currentTimeMillis())
                    }
                )

            return true;

        } catch (e: Exception) {
            Log.e("UpdateService", "Network or database error for ${symbol.name}: ${e.message}")
            false
        }
    }


    private fun createNotification(text: String, progress: Int, max: Int = 100): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Updating Symbols")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setContentIntent(pendingIntent)
            .setProgress(max, progress, false)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String, progress: Int, current: Int, total: Int) {
        val notification = createNotification("$text ($current/$total)", progress)
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun broadcastProgress(progress: Int, symbolName: String, current: Int, total: Int) {
        val intent = Intent(BROADCAST_UPDATE_PROGRESS)
        intent.setPackage(packageName)
        intent.putExtra(EXTRA_PROGRESS, progress)
        intent.putExtra(EXTRA_SYMBOL_NAME, symbolName)
        intent.putExtra(EXTRA_CURRENT, current)
        intent.putExtra(EXTRA_TOTAL, total)
        sendBroadcast(intent)
        Log.d("Service", "Sent broadcast: $progress%")
    }

    private fun broadcastComplete(successCount: Int, failureCount: Int, failedSymbols: List<String>) {
        val intent = Intent(BROADCAST_UPDATE_COMPLETE)
        intent.setPackage(packageName)
        intent.putExtra("success_count", successCount)
        intent.putExtra("failure_count", failureCount)
        intent.putStringArrayListExtra("failed_symbols", ArrayList(failedSymbols))
        sendBroadcast(intent)
    }

    private fun broadcastCancelled(successCount: Int, failureCount: Int, processedCount: Int) {
        val intent = Intent(BROADCAST_UPDATE_CANCELLED)
        intent.setPackage(packageName)
        intent.putExtra("success_count", successCount)
        intent.putExtra("failure_count", failureCount)
        intent.putExtra("processed_count", processedCount)
        sendBroadcast(intent)
    }

    private fun broadcastError(errorMessage: String) {
        val intent = Intent(BROADCAST_UPDATE_ERROR)
        intent.setPackage(packageName)
        intent.putExtra("error_message", errorMessage)
        sendBroadcast(intent)
    }

    private fun stopUpdateProcess() {
        isUpdating = false
        stopForeground(true)
        stopSelf()
    }
}