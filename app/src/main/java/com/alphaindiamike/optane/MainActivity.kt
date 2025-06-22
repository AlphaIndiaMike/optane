package com.alphaindiamike.optane

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.MenuItem
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.BroadcastReceiver
import android.net.Uri
import android.widget.ProgressBar
import android.widget.Toast
import android.app.ActivityManager
import android.content.Context
import android.content.IntentFilter
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.textfield.TextInputEditText
import com.alphaindiamike.optane.adapter.SymbolAdapter
import com.alphaindiamike.optane.database.DatabaseCapRepository
import com.alphaindiamike.optane.database.DatabaseItr
import com.alphaindiamike.optane.database.entities.MainAssetEntity
import com.alphaindiamike.optane.databinding.ActivityMainBinding
import com.alphaindiamike.optane.model.Symbol
import com.alphaindiamike.optane.services.UpdateAllSymbolsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log


class MainActivity : OptaneActivityBase<ActivityMainBinding>() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var fabAdd: FloatingActionButton
    private lateinit var editTextSearch: TextInputEditText
    private lateinit var symbolAdapter: SymbolAdapter
    private lateinit var textViewEmpty: TextView
    private lateinit var layoutEmptyState: LinearLayout
    private lateinit var repository: DatabaseCapRepository
    private var updateReceiver: BroadcastReceiver? = null
    private var progressDialog: AlertDialog? = null

    // All symbols from database - no more hardcoded symbols
    private val allSymbols = mutableListOf<Symbol>()
    // Filtered symbols based on search
    private val filteredSymbols = mutableListOf<Symbol>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)

        setupDatabase()
        setupViews()
        setupRecyclerView()
        setupSearchFunctionality()
        setupFabs()

        // Load symbols from database instead of showing hardcoded data
        loadSymbolsFromDatabase()
    }

    override fun getViewBinding(): ActivityMainBinding {
        return ActivityMainBinding.inflate(layoutInflater)
    }

    override fun getMenuId(): Int {
        return R.menu.menu_main
    }

    override fun getToolbarTitle(): String {
        return "Asset list"
    }

    private fun setupDatabase() {
        val database = DatabaseItr.getDatabase(this)
        repository = DatabaseCapRepository(database)
    }


    private fun setupViews() {
        recyclerView = findViewById(R.id.recyclerViewSymbols)
        fabAdd = findViewById(R.id.fabAdd)
        //fabRefresh = findViewById(R.id.fabRefresh)
        editTextSearch = findViewById(R.id.editTextSearch)
        layoutEmptyState = findViewById(R.id.layoutEmptyState)
        textViewEmpty = findViewById(R.id.textViewEmpty)
    }

    private fun setupRecyclerView() {
        symbolAdapter = SymbolAdapter(filteredSymbols, onItemClick = { symbol ->
            // Navigate to SymbolActivity
            val intent = Intent(this, SymbolActivity::class.java)
            intent.putExtra("symbol_name", symbol.name)
            intent.putExtra("fk_id", symbol.id)
            intent.putExtra("symbol_id", symbol.exchangeId)
            startActivity(intent)
        });

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = symbolAdapter
    }

    private fun setupSearchFunctionality() {
        editTextSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                updateFilteredSymbols(s.toString())
            }

            override fun afterTextChanged(s: Editable?) {}
        })
    }

    private fun setupFabs() {
        fabAdd.setOnClickListener {
            val intent = Intent(this, AddSymbolActivity::class.java)
            startActivity(intent)
        }

        /*
        fabRefresh.setOnClickListener {
            refreshSymbols()
        }*/
    }

    private fun loadSymbolsFromDatabase() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val mainAssets = repository.getAllAssets()

                // Convert MainAsset entities to Symbol objects
                val symbols = mainAssets.map { mainAsset ->
                    Symbol(
                        id = mainAsset.id.toString(),
                        name = mainAsset.name,
                        exchangeId = mainAsset.exchangeId.toString(),
                        lastUpdate = mainAsset.lastUpdate ?: 0L
                    )
                }

                // Switch to main thread for UI updates
                withContext(Dispatchers.Main) {
                    allSymbols.clear()
                    allSymbols.addAll(symbols)

                    val currentQuery = editTextSearch.text.toString()
                    updateFilteredSymbols(currentQuery)
                    updateEmptyState()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    // Handle error - could show a Toast
                    updateEmptyState()
                }
            }
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    private fun updateFilteredSymbols(query: String) {
        filteredSymbols.clear()

        if (query.isEmpty()) {
            filteredSymbols.addAll(allSymbols)
        } else {
            val searchQuery = query.lowercase()
            filteredSymbols.addAll(
                allSymbols.filter { symbol ->
                    symbol.name.lowercase().contains(searchQuery) ||
                            symbol.exchangeId.lowercase().contains(searchQuery)
                }
            )
        }

        symbolAdapter.notifyDataSetChanged()
    }

    private fun updateEmptyState() {
        if (allSymbols.isEmpty()) {
            // Show empty state for no symbols in database
            recyclerView.visibility = View.GONE
            layoutEmptyState.visibility = View.VISIBLE
            textViewEmpty.text = "No symbols added yet.\nTap + to add your first symbol."
        } else if (filteredSymbols.isEmpty()) {
            // Show empty state for search with no results
            recyclerView.visibility = View.GONE
            layoutEmptyState.visibility = View.VISIBLE
            textViewEmpty.text = "No symbols match your search."
        } else {
            // Show symbols
            recyclerView.visibility = View.VISIBLE
            layoutEmptyState.visibility = View.GONE
        }
    }

    override fun onResume() {
        super.onResume()
        loadSymbolsFromDatabase()
        //val currentQuery = editTextSearch.text.toString()
        //updateFilteredSymbols(currentQuery)
        // If service is running when we come back to activity, show dialog again
        if (isUpdateServiceRunning()) {
            registerUpdateReceiver()
            showProgressDialog()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == ADD_SYMBOL_REQUEST_CODE && resultCode == RESULT_OK) {
            // Reload symbols when returning from AddSymbolActivity
            loadSymbolsFromDatabase()
        }
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == IMPORT_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            val uri = data?.data
            if (uri != null) {
                processSelectedFile(uri)
            } else {
                showImportError("No file selected")
            }
        }
    }

    companion object {
        private const val ADD_SYMBOL_REQUEST_CODE = 1001
        private const val IMPORT_REQUEST_CODE = 1001
    }

    override fun handleMenuItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.action_exit -> {
                handleExit()
                return true
            }
            R.id.action_import -> {
                handleImport()
                return true
            }
            R.id.action_report -> {
                handleReport()
                return true
            }
            R.id.action_update_all -> {
                handleUpdateAll()
                return true
            }
        }
        return false
    }

    private fun handleExit() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Exit Application")
        builder.setMessage("Are you sure you want to exit?")
        builder.setPositiveButton("Exit") { dialog, which ->
            finishAffinity()
        }
        builder.setNegativeButton("Cancel") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun handleImport() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "application/json"
        intent.addCategory(Intent.CATEGORY_OPENABLE)

        try {
            startActivityForResult(intent, IMPORT_REQUEST_CODE)
        } catch (e: ActivityNotFoundException) {
            val builder = AlertDialog.Builder(this)
            builder.setTitle("Import Failed")
            builder.setMessage("No file manager found to select files")
            builder.setPositiveButton("OK") { dialog, which ->
                dialog.dismiss()
            }
            builder.show()
        }
    }

    private fun processSelectedFile(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            if (inputStream != null) {
                val content = inputStream.bufferedReader().use { reader ->
                    reader.readText()
                }

                if (isValidJson(content)) {
                    onJsonImported(content)
                } else {
                    showImportError("Selected file is not valid JSON")
                }
            } else {
                showImportError("Cannot read selected file")
            }
        } catch (e: Exception) {
            showImportError("Error reading file: ${e.message}")
        }
    }

    private fun isValidJson(content: String): Boolean {
        return try {
            org.json.JSONObject(content)
            true
        } catch (e: Exception) {
            try {
                org.json.JSONArray(content)
                true
            } catch (e2: Exception) {
                false
            }
        }
    }

    private fun showImportError(message: String) {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Import Failed")
        builder.setMessage(message)
        builder.setPositiveButton("OK") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun insertSymbolToDatabase(symbolName: String, symbolId: String, onSuccess: () -> Unit, onError: (String) -> Unit) {
        val mainAsset = MainAssetEntity(
            id = 0,
            name = symbolName,
            exchangeId = symbolId,
            lastUpdate = null
        )

        CoroutineScope(Dispatchers.IO).launch {
            try {
                repository.insertAssetSkipExisting(mainAsset)
                withContext(Dispatchers.Main) {
                    onSuccess()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onError(e.message ?: "Unknown error")
                }
            }
        }
    }

    private fun onJsonImported(jsonContent: String) {
        try {
            val jsonArray = org.json.JSONArray(jsonContent)
            var successCount = 0
            var errorCount = 0
            val totalCount = jsonArray.length()

            for (i in 0 until jsonArray.length()) {
                val jsonObject = jsonArray.getJSONObject(i)
                val symbolName = jsonObject.getString("symbol")
                val symbolId = jsonObject.getString("ID")

                insertSymbolToDatabase(
                    symbolName = symbolName,
                    symbolId = symbolId,
                    onSuccess = {
                        successCount++
                        if (successCount + errorCount == totalCount) {
                            showImportResults(successCount, errorCount)
                        }
                    },
                    onError = { errorMessage ->
                        errorCount++
                        if (successCount + errorCount == totalCount) {
                            showImportResults(successCount, errorCount)
                        }
                    }
                )
            }
        } catch (e: Exception) {
            showImportError("Invalid JSON format: ${e.message}")
        }
    }

    private fun showImportResults(successCount: Int, errorCount: Int) {
        val message = "Import completed: $successCount successful, $errorCount failed"
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun handleReport() {
        // TODO: Implement report functionality
    }

    private fun handleUpdateAll() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Update All Symbols")
        builder.setMessage("This will update all symbols from the network. Continue?")
        builder.setPositiveButton("Start") { dialog, which ->
            tryStartUpdateService()
        }
        builder.setNegativeButton("Cancel") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun tryStartUpdateService() {
        try {
            registerUpdateReceiver()
            // Add a small delay to ensure receiver is registered
            Handler(Looper.getMainLooper()).postDelayed({
                showProgressDialog()

                val intent = Intent(this, UpdateAllSymbolsService::class.java)
                intent.action = UpdateAllSymbolsService.ACTION_START_UPDATE
                startService(intent)
            }, 100) // 100ms delay


        } catch (e: SecurityException) {
            dismissProgressDialog()
            showPermissionDialog()
        } catch (e: Exception) {
            dismissProgressDialog()
            showUpdateError("Failed to start update service: ${e.message}")
        }
    }

    private fun showPermissionDialog() {
        val builder = AlertDialog.Builder(this)
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
                    UpdateAllSymbolsService.BROADCAST_UPDATE_PROGRESS -> {
                        val progress = intent.getIntExtra(UpdateAllSymbolsService.EXTRA_PROGRESS, 0)
                        val symbolName = intent.getStringExtra(UpdateAllSymbolsService.EXTRA_SYMBOL_NAME) ?: ""
                        val current = intent.getIntExtra(UpdateAllSymbolsService.EXTRA_CURRENT, 0)
                        val total = intent.getIntExtra(UpdateAllSymbolsService.EXTRA_TOTAL, 0)

                        updateProgressDialog(symbolName, progress, current, total)
                    }
                    UpdateAllSymbolsService.BROADCAST_UPDATE_COMPLETE -> {
                        val successCount = intent.getIntExtra("success_count", 0)
                        val failureCount = intent.getIntExtra("failure_count", 0)
                        val failedSymbols = intent.getStringArrayListExtra("failed_symbols") ?: arrayListOf()

                        dismissProgressDialog()
                        showUpdateResults(successCount, failureCount, failedSymbols, false)
                    }
                    UpdateAllSymbolsService.BROADCAST_UPDATE_CANCELLED -> {
                        val successCount = intent.getIntExtra("success_count", 0)
                        val failureCount = intent.getIntExtra("failure_count", 0)
                        val processedCount = intent.getIntExtra("processed_count", 0)

                        dismissProgressDialog()
                        showCancelledResults(successCount, failureCount, processedCount)
                    }
                    UpdateAllSymbolsService.BROADCAST_UPDATE_ERROR -> {
                        val error = intent.getStringExtra("error_message") ?: "Unknown error"
                        dismissProgressDialog()
                        showUpdateError(error)
                    }
                }
            }
        }

        val filter = IntentFilter()
        filter.addAction(UpdateAllSymbolsService.BROADCAST_UPDATE_PROGRESS)
        filter.addAction(UpdateAllSymbolsService.BROADCAST_UPDATE_COMPLETE)
        filter.addAction(UpdateAllSymbolsService.BROADCAST_UPDATE_CANCELLED)
        filter.addAction(UpdateAllSymbolsService.BROADCAST_UPDATE_ERROR)
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
        builder.setTitle("Updating Symbols")
        builder.setCancelable(false)
        builder.setNegativeButton("Cancel") { dialog, which ->
            stopUpdateService()
        }
        builder.setOnDismissListener { dialog ->
            // If dialog is dismissed for any reason, stop the service
            stopUpdateService()
        }

        progressDialog = builder.create()
        progressDialog?.show()
    }

    private fun updateProgressDialog(symbolName: String, progress: Int, current: Int, total: Int) {
        progressDialog?.let { dialog ->
            val progressText = dialog.findViewById<TextView>(R.id.progressText)
            val progressBar = dialog.findViewById<ProgressBar>(R.id.progressBar)

            progressText?.text = "Updating $symbolName ($current/$total)"
            progressBar?.progress = progress
        }
    }

    private fun dismissProgressDialog() {
        progressDialog?.setOnDismissListener(null) // Prevent triggering stopUpdateService again
        progressDialog?.dismiss()
        progressDialog = null
    }

    private fun stopUpdateService() {
        val intent = Intent(this, UpdateAllSymbolsService::class.java)
        intent.action = UpdateAllSymbolsService.ACTION_STOP_UPDATE
        startService(intent)
        dismissProgressDialog()
    }

    private fun showUpdateResults(successCount: Int, failureCount: Int, failedSymbols: List<String>, wasCancelled: Boolean) {
        val title = if (wasCancelled) "Update Cancelled" else "Update Complete"
        val message = buildString {
            appendLine("Successfully updated: $successCount symbols")

            if (failureCount.toInt() > 0) {
                appendLine("Failed to update: $failureCount symbols")
            }

            if (failedSymbols.isNotEmpty()) {
                appendLine("\nFailed symbols:")
                failedSymbols.take(5).forEach { symbolName ->
                    appendLine("• $symbolName")
                }
                if (failedSymbols.size > 5) {
                    appendLine("• ... and ${failedSymbols.size - 5} more")
                }
            }
        }

        val builder = AlertDialog.Builder(this)
        builder.setTitle(title)
        builder.setMessage(message)
        builder.setPositiveButton("OK") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
        loadSymbolsFromDatabase()
    }

    private fun showCancelledResults(successCount: Int, failureCount: Int, processedCount: Int) {
        val message = "Update was cancelled after processing $processedCount symbols.\n\n" +
                "Successfully updated: $successCount\n" +
                "Failed: $failureCount"

        val builder = AlertDialog.Builder(this)
        builder.setTitle("Update Cancelled")
        builder.setMessage(message)
        builder.setPositiveButton("OK") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun showUpdateError(message: String) {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Update Failed")
        builder.setMessage(message)
        builder.setPositiveButton("OK") { dialog, which ->
            dialog.dismiss()
        }
        builder.show()
    }

    private fun isUpdateServiceRunning(): Boolean {
        val manager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        for (service in manager.getRunningServices(Integer.MAX_VALUE)) {
            if (UpdateAllSymbolsService::class.java.name == service.service.className) {
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
    }


}