package com.alphaindiamike.optane.services

import android.app.AlertDialog
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
import com.alphaindiamike.optane.algorithms.AlgorithmFactory
import com.alphaindiamike.optane.algorithms.AlgorithmType
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Calculations
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ProbabilityCalculatorService: Service() {
    private val binder = UpdateBinder()
    private var isWorkInProgress = false
    private lateinit var repository: DatabaseCapRepository

    private var symbolName : String = "Unknown";
    private var symbolId : String = "N/A";
    private var lowerBand : Double = 0.0;
    private var upperBand : Double = 0.0;
    private var days : Int = 0;
    private var fkId : Long = 0L;

    companion object {
        const val ACTION_START_CALCULATION = "START_CALCULATION"
        const val ACTION_STOP_CALCULATION = "STOP_CALCULATION"
        const val NOTIFICATION_ID = 1011
        const val CHANNEL_ID = "update_channel"

        const val EXTRA_PROGRESS = "progress"
        const val EXTRA_CALC_NAME = "calculation_name"
        const val EXTRA_TOTAL = "total"
        const val EXTRA_CURRENT = "current"

        const val BROADCAST_UPDATE_PROGRESS = "com.alphaindiamike.optane.UPDATE_PROGRESS"
        const val BROADCAST_UPDATE_COMPLETE = "com.alphaindiamike.optane.UPDATE_COMPLETE"
        const val BROADCAST_UPDATE_ERROR = "com.alphaindiamike.optane.UPDATE_ERROR"
        const val BROADCAST_UPDATE_CANCELLED = "com.alphaindiamike.optane.UPDATE_CANCELLED"
    }

    private fun getIntentData(intent: Intent)
    {
        symbolName = intent.getStringExtra("symbol_name") ?: "Unknown"
        symbolId = intent.getStringExtra("symbol_id") ?: "N/A"
        lowerBand = intent.getDoubleExtra("lower_band", 0.0)
        upperBand = intent.getDoubleExtra("upper_band", 0.0)
        days = intent.getIntExtra("days", 0)
        fkId = try {
            intent.getLongExtra("fk_id", 0L)
        } catch (e: NumberFormatException) {
            0L
        }
    }

    inner class UpdateBinder : Binder() {
        fun getService(): ProbabilityCalculatorService = this@ProbabilityCalculatorService
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
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        intent?.let {
            getIntentData(it)  // Pass intent explicitly
        }

        setupDatabase()
        createNotificationChannel()

        when (intent?.action) {
            ProbabilityCalculatorService.Companion.ACTION_START_CALCULATION -> {
                if (!isWorkInProgress) {
                    statCalculationProcess()
                }
            }
            ProbabilityCalculatorService.Companion.ACTION_STOP_CALCULATION -> {
                stopCalculationProcess()
                return START_NOT_STICKY // Important: don't restart
            }
        }
        return START_STICKY
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            ProbabilityCalculatorService.Companion.CHANNEL_ID,
            "Probability calculation",
            NotificationManager.IMPORTANCE_LOW
        )
        channel.description = "Probability calculation feedback"

        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }

    private fun statCalculationProcess() {
        isWorkInProgress = true
        val notification = createNotification("Starting calculation...", 0, 0)
        startForeground(NOTIFICATION_ID, notification)
        CoroutineScope(Dispatchers.IO).launch {
            try {
                if (!isWorkInProgress) {
                    broadcastCancelled()
                    return@launch
                }
                Log.d("Service", "Started calculations..")
                performCalculations()
            } catch (e: Exception) {
                broadcastError("Failed to load symbols: ${e.message}")
            } finally {
                stopSelf()
            }
        }
    }

    private fun createNotification(text: String, progress: Int, max: Int = 100): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, ProbabilityCalculatorService.Companion.CHANNEL_ID)
            .setContentTitle("Calculating probabilities...")
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
        notificationManager.notify(ProbabilityCalculatorService.Companion.NOTIFICATION_ID, notification)
    }

    private fun broadcastProgress(progress: Int, current: Int, total: Int, calcName: String) {
        val intent = Intent(BROADCAST_UPDATE_PROGRESS)
        intent.setPackage(packageName)
        intent.putExtra(EXTRA_PROGRESS, progress)
        intent.putExtra(EXTRA_CALC_NAME, calcName)
        intent.putExtra(EXTRA_CURRENT, current)
        intent.putExtra(EXTRA_TOTAL, total)
        sendBroadcast(intent)
        updateNotification("Calculating $calcName",progress,current,total)
        Log.d("Service", "Sent broadcast: $progress%")
    }

    private fun broadcastComplete(result : String) {
        val intent = Intent(BROADCAST_UPDATE_COMPLETE)
        intent.setPackage(packageName)
        intent.putExtra("result", result)
        sendBroadcast(intent)

        // Cancel the notification when complete
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.cancel(NOTIFICATION_ID)
        stopForeground(true) // This should also remove notification
    }

    private fun broadcastCancelled() {
        val intent = Intent(BROADCAST_UPDATE_CANCELLED)
        intent.setPackage(packageName)
        sendBroadcast(intent)

        // Cancel the notification when complete
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.cancel(NOTIFICATION_ID)
        stopForeground(true) // This should also remove notification
    }

    private fun broadcastError(errorMessage: String) {
        val intent = Intent(ProbabilityCalculatorService.Companion.BROADCAST_UPDATE_ERROR)
        intent.setPackage(packageName)
        intent.putExtra("error_message", errorMessage)
        sendBroadcast(intent)

        // Cancel the notification when complete
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.cancel(NOTIFICATION_ID)
        stopForeground(true) // This should also remove notification
    }

    private fun stopCalculationProcess() {
        Log.d("Service", "Stopping calculation process...")
        isWorkInProgress = false
        // Cancel notification
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.cancel(NOTIFICATION_ID)

        // Stop foreground and self
        stopForeground(true)
        stopSelf()

        // Broadcast cancellation
        broadcastCancelled()

        Log.d("Service", "Service stop process completed")
    }

    private fun performCalculations()
    {
        var calculationReport = "";
        if (0L == fkId)
        {
            broadcastError("Error with the intent - invalid ID!")
            return;
        }
        // Load data from database in background thread
        CoroutineScope(Dispatchers.IO).launch (
            // Coroutine Context
            Dispatchers.IO,
            // Suspend function block
            block = {
            try {
                    val timeSeriesData = withContext( // Call
                        //Coroutine Context
                        Dispatchers.IO,
                        //Suspend function
                        block = {
                            repository.getTimeSeriesForAsset(fkId).sortedBy(
                                selector =
                                    // Selector function
                                    { it.date }
                            )
                        }
                    )

                    if (timeSeriesData.isNotEmpty()) {
                        calculationReport = generateReportText(timeSeriesData);
                        if (isWorkInProgress) {
                            broadcastComplete(calculationReport)
                        }
                    } else {
                        calculationReport = """
    No data available for $symbolName ($symbolId)
    """.trimIndent()
                        broadcastError(calculationReport)
                    }
                } catch (e: Exception) {
                    Log.e("SymbolActivity", "Error loading chart data", e)
                    calculationReport = """
    Error loading data ... 
                """
                broadcastError(calculationReport)
                }
            }
        )
    }

    private suspend fun generateReportText(input: List<TimeSeriesEntity>) : String {
        val calcParam = Calculations(
            name = symbolName,
            lowerPriceBand = lowerBand,
            upperPriceBand = upperBand,
            daysPrediction = days,
            timeSeries = input
        );

        broadcastProgress(1,1,5,"REGIME_SWITCHING_FORECASTER");
        val reportRSF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.REGIME_SWITCHING_FORECASTER)
            .calculate(calcParam)
        /* Don't work */
        /*
        val reportMCB = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.MONTE_CARLO_BASIC)
            .calculate(calcParam)
        val reportMCA = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.MONTE_CARLO_ADVANCED)
            .calculate(calcParam) */
        broadcastProgress(20,2,5,"PROBABILISTIC_FORECASTER");
        val reportPFB = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.PROBABILISTIC_FORECASTER)
            .calculate(calcParam)
        /* val reportQTF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.QUANTILE_TRANSFORMER_FORECASTER)
            .calculate(calcParam)
        val reportTF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.TRANSFORMER_FORECASTER)
            .calculate(calcParam)
        val reportMEF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.META_ENSEMBLE_FORECASTER)
            .calculate(calcParam)*/
        broadcastProgress(40,3,5,"JUMP_DIFFUSION_FORECASTER");
        val reportJDF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.JUMP_DIFFUSION_FORECASTER)
            .calculate(calcParam)
        broadcastProgress(60,4,5,"GARCH_FORECASTER");
        val reportGF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.GARCH_FORECASTER)
            .calculate(calcParam)
        /* val reportEF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.ENSEMBLE_FORECASTER)
            .calculate(calcParam) */
        broadcastProgress(80,5,5,"BLACK_SCHOLES_FORECASTER");
        val reportBSF = AlgorithmFactory()
            .createAlgorithm(AlgorithmType.BLACK_SCHOLES_FORECASTER)
            .calculate(calcParam)
        return """
Asset Analysis Report
=====================

Symbol: $symbolName ($symbolId)
Analysis Date: ${
            java.text.SimpleDateFormat(
                "dd.MM.yyyy HH:mm",
                java.util.Locale.getDefault()
            ).format(java.util.Date())
        }

Price Band Analysis:
• Upper Band: ${String.format("%.2f", upperBand)}
• Lower Band: ${String.format("%.2f", lowerBand)}

• Target Period: $days days

Probabilistic Forecaster (Finance School)
=====================
$reportPFB

Regime Switching Forecast (Markov Chains)
=====================
$reportRSF

Jump-Diffusion Forecaster 
=====================
$reportJDF

GARCH Forecaster
=====================
$reportGF

Black-Scholes Forecaster
=====================
$reportBSF


""".trimIndent()

    }

}