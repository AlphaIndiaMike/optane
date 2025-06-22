package com.alphaindiamike.optane

import android.app.AlertDialog
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.annotation.MenuRes
import androidx.appcompat.app.AppCompatActivity
import androidx.viewbinding.ViewBinding

abstract class OptaneActivityBase<VB: ViewBinding> : AppCompatActivity(){
    private var _binding: VB? = null
    protected val binding get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = getViewBinding()
        setContentView(binding.root)

        setupToolbar()
        initializeViews()
    }

    override fun onDestroy() {
        super.onDestroy()
        _binding = null
    }

    abstract fun getViewBinding(): VB

    @MenuRes
    open fun getMenuId(): Int {
        return 0
    }

    open fun getToolbarTitle(): String? {
        return "Optane: probabilistic price calculator"
    }

    open fun shouldShowBackButton(): Boolean {
        return false
    }

    open fun initializeViews() {
        // Override in child activities
    }

    private fun setupToolbar() {
        val toolbar = findViewById<com.google.android.material.appbar.MaterialToolbar>(R.id.toolbar)
        if (toolbar != null) {
            setSupportActionBar(toolbar)

            val actionBar = supportActionBar
            if (actionBar != null) {
                actionBar.setDisplayHomeAsUpEnabled(shouldShowBackButton())
                actionBar.setDisplayShowHomeEnabled(shouldShowBackButton())

                val title = getToolbarTitle()
                if (title != null) {
                    actionBar.title = title
                }
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        val menuId = getMenuId()
        if (menuId != 0) {
            menuInflater.inflate(menuId, menu)
            return true
        }
        return super.onCreateOptionsMenu(menu)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> {
                handleBackNavigation()
                return true
            }
            else -> {
                if (handleMenuItemSelected(item)) {
                    return true
                }
            }
        }
        return super.onOptionsItemSelected(item)
    }

    open fun handleMenuItemSelected(item: MenuItem): Boolean {
        AlertDialog.Builder(this)
            .setTitle("Menu pressed")
            .setMessage("Option ${item.title.toString()}")
            .show()
        return false
    }

    open fun handleBackNavigation() {
        onBackPressedDispatcher.onBackPressed()
    }
}