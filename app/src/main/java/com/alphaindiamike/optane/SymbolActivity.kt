package com.alphaindiamike.optane

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.app.AlertDialog
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.alphaindiamike.optane.database.DatabaseCapRepository
import com.alphaindiamike.optane.database.DatabaseItr
import com.alphaindiamike.optane.database.entities.TimeSeriesEntity
import com.alphaindiamike.optane.model.Symbol
import com.alphaindiamike.optane.network.NetworkCapRepository
import com.alphaindiamike.optane.network.implementations.LsTcDownload
import com.alphaindiamike.optane.network.model.NetworkException
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.components.YAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*
import android.content.Context
import android.content.res.Configuration
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.components.MarkerView
import com.github.mikephil.charting.utils.MPPointF

class SymbolActivity : AppCompatActivity() {

    private lateinit var textAssetTitle: TextView
    private lateinit var textLastUpdate: TextView
    private lateinit var editLowerPriceBand: TextInputEditText
    private lateinit var editUpperPriceBand: TextInputEditText
    private lateinit var editDaysPrediction: TextInputEditText
    private lateinit var btnCalculateProbability: Button
    private lateinit var repositoryDatabaseLink: DatabaseCapRepository
    private lateinit var lineChart: LineChart
    private lateinit var fabRefresh: FloatingActionButton
    private lateinit var fabDelete: FloatingActionButton

    private var symbolName: String = ""
    private var symbolId: String = ""
    private var fkId: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_symbol)

        setupDatabase()
        getIntentData()
        setupViews()
        setupClickListeners()
        updateUI()
        setupChart()
        refreshChartData()
    }

    private fun setupDatabase() {
        val database = DatabaseItr.getDatabase(this)
        repositoryDatabaseLink = DatabaseCapRepository(database)
    }

    private fun getIntentData() {
        symbolName = intent.getStringExtra("symbol_name") ?: ""
        symbolId = intent.getStringExtra("symbol_id") ?: ""
        fkId = try {
            intent.getStringExtra("fk_id")?.toLong() ?: 0L
        } catch (e: NumberFormatException) {
            0L
        }
    }

    private fun setupViews() {
        textAssetTitle = findViewById(R.id.textAssetTitle)
        textLastUpdate = findViewById(R.id.textLastUpdate)
        editLowerPriceBand = findViewById(R.id.editLowerPriceBand)
        editUpperPriceBand = findViewById(R.id.editUpperPriceBand)
        editDaysPrediction = findViewById(R.id.editDaysPrediction)
        btnCalculateProbability = findViewById(R.id.btnCalculateProbability)
        lineChart = findViewById(R.id.lineChart)
        fabRefresh = findViewById(R.id.fabRefresh)
        fabDelete = findViewById(R.id.fabDelete)
    }

    private fun setupClickListeners() {
        btnCalculateProbability.setOnClickListener {
            calculateProbability()
        }

        fabRefresh.setOnClickListener {
            refreshDataFromNetwork()
        }

        fabDelete.setOnClickListener {
            // Implement delete functionality
            showDeleteConfirmation()
        }
    }

    private fun refreshDataFromNetwork() {
        // Disable refresh button to prevent multiple clicks
        fabRefresh.isEnabled = false
        textLastUpdate.text = "Refreshing data..."

        lifecycleScope.launch {
            try {
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
                withContext(
                    // Coroutine Context
                    Dispatchers.IO,
                    // Suspend function block
                    block = {
                        repositoryDatabaseLink.deleteAllTimeSeriesForAsset(fkId)
                        repositoryDatabaseLink.insertTimeSeriesData(timeSeriesEntities)
                        repositoryDatabaseLink.updateAssetLastUpdate(fkId, System.currentTimeMillis())
                    }
                )

                // Step 4: Update UI and reload chart
                updateLastUpdateTime(System.currentTimeMillis())

                loadChartData()
                updateUI()
                Toast.makeText(this@SymbolActivity, "Data refreshed successfully", Toast.LENGTH_SHORT).show()

            } catch (e: NetworkException) {
                textLastUpdate.text = "Last update: Network error"
                showErrorDialog("Network Error", e.message ?: "Failed to download data")
            } catch (e: Exception) {
                textLastUpdate.text = "Last update: Error occurred"
                showErrorDialog("Error", "Failed to refresh data: ${e.message}")
            } finally {
                fabRefresh.isEnabled = true
            }
        }
    }

    private fun showErrorDialog(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", null)
            .setNeutralButton("Retry") { _, _ ->
                refreshDataFromNetwork()
            }
            .show()
    }

    private fun setupChart() {
        lineChart.apply {
            // Get system colors based on theme
            val isDarkMode = resources.configuration.uiMode and
                    Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES

            val backgroundColor = if (isDarkMode) Color.BLACK else Color.WHITE
            val textColor = if (isDarkMode) Color.WHITE else Color.LTGRAY
            val gridColor = if (isDarkMode) Color.DKGRAY else Color.LTGRAY

            // Apply system colors
            setBackgroundColor(backgroundColor)
            setGridBackgroundColor(backgroundColor)

            // Configure X axis with system colors
            xAxis.apply(block = {
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(true)
                setGridColor(gridColor)
                setTextColor(textColor)
                textSize = 10f
                setLabelCount(6, false)

                valueFormatter = object : ValueFormatter() {
                    private val dateFormat = SimpleDateFormat("dd.MM.yy", Locale.getDefault())
                    override fun getFormattedValue(value: Float): String {
                        // Convert nanoseconds to milliseconds
                        val milliseconds = value.toLong() / 1_000_000
                        return dateFormat.format(Date(milliseconds))
                    }
                }
            });

            // Configure left Y axis with system colors
            axisLeft.apply( block ={
                setDrawGridLines(true)
                textSize = 10f
                setGridColor(gridColor)
                setTextColor(textColor)
                setPosition(YAxis.YAxisLabelPosition.OUTSIDE_CHART)

                valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        return String.format("%.2f", value)
                    }
                }
            })

            axisRight.isEnabled = false

            // Legend with system colors
            legend.apply {
                isEnabled = true
                textSize = 12f
                form = Legend.LegendForm.LINE
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
            }

            animateX(1000)

            // Add custom marker
            val marker = CustomMarkerView(this@SymbolActivity, R.layout.component_custom_marker_view)
            marker.chartView = this
            this.marker = marker

            // Disable the default marker info
            setDrawMarkers(true)
        }
    }


    private fun showEmptyChart() {
        lineChart.apply {
            clear()
            setNoDataText("No price data available for $symbolName\n\nTap the refresh button to load data")
            setNoDataTextColor(Color.GRAY)
            invalidate()
        }

        // Also update the last update text to reflect empty state
        textLastUpdate.text = "Last update: No data available - Please refresh"
    }

    private fun loadChartData() {
        if (fkId == 0L) {
            // Handle case where database ID is not provided
            showEmptyChart()
            return
        }

        // Load data from database in background thread
        lifecycleScope.launch {
            try {
                val timeSeriesData = withContext( // Call
                    //Coroutine Context
                    Dispatchers.IO,
                    //Suspend function
                    block = {
                        repositoryDatabaseLink.getTimeSeriesForAsset(fkId).sortedBy(
                            selector =
                                // Selector function
                                { it.date }
                        )
                    }
                )

                if (timeSeriesData.isNotEmpty()) {
                    updateChart(timeSeriesData)
                    updateLastUpdateTime(timeSeriesData.maxByOrNull { it.date }?.date)
                } else {
                    showEmptyChart()
                }
            } catch (e: Exception) {
                Log.e("SymbolActivity", "Error loading chart data", e)
                showEmptyChart()
            }
        }
    }

    private fun updateChart(timeSeriesData: List<TimeSeriesEntity>) {
        val entries = timeSeriesData.map {
            Entry(it.date.toFloat(), it.price.toFloat())
        }

        val dataSet = LineDataSet(entries, symbolName).apply {
            // Line styling for professional trading look
            color = Color.rgb(33, 150, 243) // Blue color
            lineWidth = 2f
            setDrawCircles(false) // Don't draw circles on data points for cleaner look
            setDrawValues(false) // Don't show values on chart

            // Fill under the line for better visual
            setDrawFilled(true)
            fillColor = Color.rgb(33, 150, 243)
            fillAlpha = 30

            // Smooth curve
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f
        }

        val lineData = LineData(dataSet)
        lineChart.apply {
            data = lineData
            invalidate() // Refresh the chart

            // Auto-fit the chart to show all data nicely
            fitScreen()
        }
    }

    private fun refreshChartData() {
        // Add loading indicator if needed
        fabRefresh.isEnabled = false

        loadChartData()

        // Re-enable FAB after a delay (you might want to do this after actual data loading)
        Handler(Looper.getMainLooper()).postDelayed({
            fabRefresh.isEnabled = true
        }, 1000)
    }


    private fun showDeleteConfirmation() {
        AlertDialog.Builder(this)
            .setTitle("Delete Asset Data")
            .setMessage("Are you sure you want to delete all data for $symbolName?")
            .setPositiveButton("Delete") { _, _ ->
                deleteAssetData()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteAssetData() {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    repositoryDatabaseLink.deleteAsset(fkId);
                }
                // Navigate back or update UI
                finish()
            } catch (e: Exception) {
                Log.e("SymbolActivity", "Error deleting asset data", e)
                // Show error message to user
            }
        }
    }

    private fun updateLastUpdateTime(lastTimestamp: Long?) {
        val sym = Symbol("","",lastUpdate = lastTimestamp ?: 0L)

        val lastUpdateText = if (lastTimestamp != null) {
            "Last data point: ${sym.getFormattedDate()}"
        } else {
            "Last update: No data available"
        }
        textLastUpdate.text = lastUpdateText
    }


    @SuppressLint("SetTextI18n")
    private fun updateUI() {
        // Load data from database in background thread
        lifecycleScope.launch {
            try {
                val timeSeriesLastDp = withContext( // Call
                    //Coroutine Context
                    Dispatchers.IO,
                    //Suspend function
                    block = {
                        repositoryDatabaseLink.getTimeSeriesForAssetLastDataPoint(fkId);
                    }
                )
                textAssetTitle.text = "Asset title: $symbolName (Last price: ${String.format("%.2f", timeSeriesLastDp.price)} EUR) "
            } catch (e: Exception) {
                textAssetTitle.text = "Asset title: $symbolName (never updated)"
            }
        }

        // Update last update date
        val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale.getDefault())
        val currentDate = dateFormat.format(Date())
        textLastUpdate.text = "Last update: $currentDate"
    }

    private fun calculateProbability() {
        val lowerBand = editLowerPriceBand.text.toString().toDoubleOrNull() ?: 0.0
        val upperBand = editUpperPriceBand.text.toString().toDoubleOrNull() ?: 0.0
        val days = editDaysPrediction.text.toString().toIntOrNull() ?: 0

        if (lowerBand > 0 && upperBand > 0 && days > 0) {
            // Navigate to ReportActivity with calculation data
            val intent = Intent(this, ReportActivity::class.java)
            intent.putExtra("symbol_name", symbolName)
            intent.putExtra("symbol_id", symbolId)
            intent.putExtra("fk_id", fkId)
            intent.putExtra("lower_band", lowerBand)
            intent.putExtra("upper_band", upperBand)
            intent.putExtra("days", days)
            startActivity(intent)
        }
    }
}

@SuppressLint("ViewConstructor")
class CustomMarkerView(context: Context, layoutResource: Int) : MarkerView(context, layoutResource) {
    private val tvContent: TextView = findViewById(R.id.tvContent)

    override fun refreshContent(e: Entry?, highlight: Highlight?) {
        if (e != null) {
            // Show only Y value formatted as price
            tvContent.text = String.format("%.2f EUR", e.y)
        }
        super.refreshContent(e, highlight)
    }

    override fun getOffset(): MPPointF {
        return MPPointF(-(width / 2f), -height.toFloat())
    }
}