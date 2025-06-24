package com.alphaindiamike.optane

import android.app.ActivityManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.widget.ProgressBar
import android.widget.TextView
import com.alphaindiamike.optane.database.DatabaseCapRepository
import com.alphaindiamike.optane.database.DatabaseItr
import com.alphaindiamike.optane.databinding.ActivityReportBinding
import com.alphaindiamike.optane.services.ProbabilityCalculatorService
import androidx.appcompat.app.AlertDialog

class ReportActivity : OptaneActivityBase<ActivityReportBinding>() {

    private lateinit var textCalculation1: TextView
    private lateinit var repositoryDatabaseLink: DatabaseCapRepository
    private var updateReceiver: BroadcastReceiver? = null
    private var progressDialog: AlertDialog? = null

    private var symbolName : String = "Unknown";
    private var symbolId : String = "N/A";
    private var lowerBand : Double = 0.0;
    private var upperBand : Double = 0.0;
    private var days : Int = 0;
    private var fkId : Long = 0L;
    private var lastPrice : Double = 0.0;
    private var timeframeMonths: Int = -1 // -1 means "All data"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_report)

        getIntentData()

        setupViews()
        setupDatabase()
        //performCalculations()
        tryStartCalculationService()
    }

    private fun getIntentData()
    {
        symbolName = intent.getStringExtra("symbol_name") ?: "Unknown"
        symbolId = intent.getStringExtra("symbol_id") ?: "N/A"
        lowerBand = intent.getDoubleExtra("lower_band", 0.0)
        upperBand = intent.getDoubleExtra("upper_band", 0.0)
        days = intent.getIntExtra("days", 0)
        try {
            fkId = intent.getLongExtra("fk_id", 0L)
        } catch (e: NumberFormatException) {
            fkId = 0L
        }
        lastPrice = intent.getDoubleExtra("last_price", 0.0)
        timeframeMonths = intent.getIntExtra("timeframe_months", -1)

    }

    private fun setupDatabase() {
        val database = DatabaseItr.getDatabase(this)
        repositoryDatabaseLink = DatabaseCapRepository(database)
    }

    override fun getViewBinding(): ActivityReportBinding {
        return ActivityReportBinding.inflate(layoutInflater)
    }

    override fun getToolbarTitle(): String {
        return "Price calculation"
    }

    private fun setupViews() {
        textCalculation1 = binding.textCalculation1
    }

    private fun tryStartCalculationService() {
        try {
            registerUpdateReceiver()
            // Add a small delay to ensure receiver is registered
            Handler(Looper.getMainLooper()).postDelayed({
                showProgressDialog()

                val intent = Intent(this, ProbabilityCalculatorService::class.java)
                intent.action = ProbabilityCalculatorService.ACTION_START_CALCULATION
                intent.putExtra("symbol_name", symbolName)
                intent.putExtra("symbol_id", symbolId)
                intent.putExtra("fk_id", fkId)
                intent.putExtra("lower_band", lowerBand)
                intent.putExtra("upper_band", upperBand)
                intent.putExtra("days", days)
                intent.putExtra("last_price",lastPrice)
                intent.putExtra("timeframe_months", timeframeMonths)
                Log.d("Activity", "About to start service with action: ${intent.action}")
                startService(intent)
                Log.d("Activity", "startService() called")
            }, 100) // 100ms delay


        } catch (e: SecurityException) {
            dismissProgressDialog()
            showPermissionDialog()
        } catch (e: Exception) {
            dismissProgressDialog()
        }
    }

    private fun showPermissionDialog() {
        val builder = androidx.appcompat.app.AlertDialog.Builder(this)
        builder.setTitle("Permission Required")
        builder.setMessage("This app needs foreground service permission to update symbols in the background. Please grant permission in Settings.")
        builder.setPositiveButton("Open Settings") { dialog, which ->
            openAppSettings()
        }
        builder.setNegativeButton("Cancel") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun openAppSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
        val uri = Uri.fromParts("package", packageName, null)
        intent.data = uri
        startActivity(intent)
    }

    private fun unregisterUpdateReceiver() {
        updateReceiver?.let {
            try {
                unregisterReceiver(it)
            } catch (e: IllegalArgumentException) {
                // Receiver was not registered
            }
        }
    }

    private fun registerUpdateReceiver() {
        updateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                Log.d("Activity", "Received: ${intent?.action}")

                when (intent?.action) {
                    ProbabilityCalculatorService.BROADCAST_UPDATE_PROGRESS -> {
                        val progress = intent.getIntExtra(ProbabilityCalculatorService.EXTRA_PROGRESS, 0)
                        val calcName = intent.getStringExtra(ProbabilityCalculatorService.EXTRA_CALC_NAME) ?: ""
                        val current = intent.getIntExtra(ProbabilityCalculatorService.EXTRA_CURRENT, 0)
                        val total = intent.getIntExtra(ProbabilityCalculatorService.EXTRA_TOTAL, 0)
                        updateProgressDialog(progress, calcName, current, total)
                    }
                    ProbabilityCalculatorService.BROADCAST_UPDATE_COMPLETE -> {
                        val result = intent.getStringExtra("result")
                        dismissProgressDialog()
                        textCalculation1.text = result
                    }
                    ProbabilityCalculatorService.BROADCAST_UPDATE_CANCELLED -> {
                        dismissProgressDialog()

                    }
                    ProbabilityCalculatorService.BROADCAST_UPDATE_ERROR -> {
                        val error = intent.getStringExtra("error_message") ?: "Unknown error"
                        textCalculation1.text = error
                        dismissProgressDialog()

                    }
                }
            }
        }

        val filter = IntentFilter()
        filter.addAction(ProbabilityCalculatorService.BROADCAST_UPDATE_PROGRESS)
        filter.addAction(ProbabilityCalculatorService.BROADCAST_UPDATE_COMPLETE)
        filter.addAction(ProbabilityCalculatorService.BROADCAST_UPDATE_CANCELLED)
        filter.addAction(ProbabilityCalculatorService.BROADCAST_UPDATE_ERROR)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(updateReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            android.app.AlertDialog.Builder(this)
                .setTitle("Android version error")
                .setMessage("Android version not supported!")
                .show()
            return;
        }
    }

    private fun showProgressDialog() {
        val dialogView = layoutInflater.inflate(R.layout.component_dialog_progress, null)

        val builder = AlertDialog.Builder(this)
        builder.setView(dialogView)
        builder.setTitle("Performing calculations...")
        builder.setCancelable(false)
        builder.setNegativeButton("Cancel") { dialog, which ->
            stopCalculationService()
        }
        builder.setOnDismissListener { dialog ->
            // If dialog is dismissed for any reason, stop the service
            stopCalculationService()
        }

        progressDialog = builder.create()
        progressDialog?.show()
    }

    private fun updateProgressDialog(progress: Int, calcName: String, current: Int, total: Int) {
        progressDialog?.let { dialog ->
            val progressText = dialog.findViewById<TextView>(R.id.progressText)
            val progressBar = dialog.findViewById<ProgressBar>(R.id.progressBar)

            progressText?.text = "$calcName ($current/$total) - $progress%"
            progressBar?.progress = progress
        }
    }

    private fun dismissProgressDialog() {
        progressDialog?.setOnDismissListener(null)
        progressDialog?.dismiss()
        progressDialog = null
    }

    private fun stopCalculationService() {
        val intent = Intent(this, ProbabilityCalculatorService::class.java)
        intent.action = ProbabilityCalculatorService.ACTION_STOP_CALCULATION
        startService(intent)
        dismissProgressDialog()

        // Stop the service if it's still running when activity is destroyed
        if (isProbabilityCalculatorServiceRunning()) {
            val stopIntent = Intent(this, ProbabilityCalculatorService::class.java)
            stopIntent.action = ProbabilityCalculatorService.ACTION_STOP_CALCULATION
            startService(stopIntent)
            Log.d("Activity", "Stopping ProbabilityCalculatorService on activity destroy")
        }
        finish()
    }

    private fun isProbabilityCalculatorServiceRunning(): Boolean {
        val manager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        for (service in manager.getRunningServices(Integer.MAX_VALUE)) {
            if (ProbabilityCalculatorService::class.java.name == service.service.className) {
                return true
            }
        }
        return false
    }

    override fun onPause() {
        super.onPause()
        // Hide dialog when leaving activity, but keep service running
        dismissProgressDialog()
        updateReceiver?.let { unregisterReceiver(it) }
        updateReceiver = null
    }

    override fun onDestroy() {
        super.onDestroy()
        updateReceiver?.let { unregisterReceiver(it) }

        // Stop the service if it's still running when activity is destroyed
        if (isProbabilityCalculatorServiceRunning()) {
            val stopIntent = Intent(this, ProbabilityCalculatorService::class.java)
            stopIntent.action = ProbabilityCalculatorService.ACTION_STOP_CALCULATION
            startService(stopIntent)
            Log.d("Activity", "Stopping ProbabilityCalculatorService on activity destroy")
        }
    }
}