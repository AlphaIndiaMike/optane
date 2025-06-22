package com.alphaindiamike.optane

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.alphaindiamike.optane.database.DatabaseCapRepository
import com.alphaindiamike.optane.database.DatabaseItr
import com.alphaindiamike.optane.database.entities.MainAssetEntity
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AddSymbolActivity : AppCompatActivity() {

    private lateinit var editSymbolName: TextInputEditText
    private lateinit var editSymbolId: TextInputEditText
    private lateinit var fabAddSymbol: FloatingActionButton
    private lateinit var repository: DatabaseCapRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_add_symbol)

        setupDatabase()
        setupViews()
        setupClickListeners()
    }

    private fun setupDatabase() {
        val database = DatabaseItr.getDatabase(this)
        repository = DatabaseCapRepository(database)
    }


    private fun setupViews() {
        editSymbolName = findViewById(R.id.editSymbolName)
        editSymbolId = findViewById(R.id.editSymbolId)
        fabAddSymbol = findViewById(R.id.fabAddSymbol)
    }

    private fun setupClickListeners() {
        fabAddSymbol.setOnClickListener {
            addSymbol()
        }
    }

    private fun addSymbol() {
        val symbolName = editSymbolName.text.toString().trim()
        val symbolId = editSymbolId.text.toString().trim()

        if (symbolName.isEmpty() || symbolId.isEmpty()) {
            Toast.makeText(this, "Please fill in both fields", Toast.LENGTH_SHORT).show()
            return
        }

        // Disable the button to prevent multiple clicks
        fabAddSymbol.isEnabled = false

        // Create MainAsset entity
        val mainAsset = MainAssetEntity(
            id = 0,
            name = symbolName,
            exchangeId = symbolId, // Default exchange for now
            lastUpdate = null
        )

        // Save to database using coroutines (conservative approach)
        CoroutineScope(Dispatchers.IO).launch {
            try {
                repository.insertAsset(mainAsset)

                // Switch back to main thread for UI updates
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@AddSymbolActivity,
                        "Symbol $symbolName added successfully!",
                        Toast.LENGTH_SHORT
                    ).show()

                    setResult(RESULT_OK) // Signal success to MainActivity
                    finish()
                }
            } catch (e: Exception) {
                // Handle database error
                withContext(Dispatchers.Main) {
                    fabAddSymbol.isEnabled = true // Re-enable button
                    Toast.makeText(
                        this@AddSymbolActivity,
                        "Error adding symbol: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }
}