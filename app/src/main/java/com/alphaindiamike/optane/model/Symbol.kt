package com.alphaindiamike.optane.model
import java.text.SimpleDateFormat
import java.util.*
import com.alphaindiamike.optane.database.entities.MainAssetEntity

data class Symbol(
    val id: String,
    val name: String,
    val exchangeId: String = "UNDEF",
    val lastUpdate: Long = System.currentTimeMillis(),
    val currentPrice: Double = 0.0,
    val lowerPriceBand: Double = 0.0,
    val upperPriceBand: Double = 0.0,
    val daysPrediction: Int = 0,
    val probability: Double = 0.0
){
    // Convert Symbol to MainAsset for database storage
    fun toMainAsset(): MainAssetEntity {
        return MainAssetEntity(
            id = id.hashCode().toLong(), // Convert string ID to Long for database
            name = name,
            exchangeId = exchangeId,
            lastUpdate = lastUpdate
        )
    }

    // UI-friendly date formatting methods
    fun getFormattedDate(isSystem: Boolean = false): String {
        if (lastUpdate == 0L) return "never updated"

        // Convert nanoseconds to milliseconds
        var timestampMillis = 0L
        if (isSystem == false) {
            timestampMillis = lastUpdate / 1_000_000
        }
        else
        {
            timestampMillis = lastUpdate
        }

        val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
        return dateFormat.format(Date(timestampMillis))
    }

    fun getFormattedDateTime(isSystem: Boolean = false): String {
        if (lastUpdate == 0L) return "never updated"

        // Convert nanoseconds to milliseconds
        var timestampMillis = 0L
        if (isSystem == false) {
            timestampMillis = lastUpdate / 1_000_000
        }
        else
        {
            timestampMillis = lastUpdate
        }
        val dateTimeFormat = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
        return dateTimeFormat.format(Date(timestampMillis))
    }

    fun getTimeAgo(): String {
        if (lastUpdate == 0L) return "never updated"
        val now = System.currentTimeMillis()
        val diff = now - lastUpdate

        return when {
            diff < 60000 -> "Just now"
            diff < 3600000 -> "${diff / 60000} minutes ago"
            diff < 86400000 -> "${diff / 3600000} hours ago"
            else -> "${diff / 86400000} days ago"
        }
    }

    companion object {
        // Convert MainAsset from database back to Symbol
        fun fromMainAsset(mainAsset: MainAssetEntity): Symbol {
            return Symbol(
                id = mainAsset.id.toString(), // Convert Long ID back to String
                name = mainAsset.name,
                exchangeId = mainAsset.exchangeId,
                lastUpdate = mainAsset.lastUpdate ?: 0L
            )
        }
    }
}
