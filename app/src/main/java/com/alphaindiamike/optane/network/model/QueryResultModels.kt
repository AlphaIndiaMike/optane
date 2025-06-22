package com.alphaindiamike.optane.network.model

// Result class for refresh operations
sealed class RefreshResult {
    data class Success(
        val recordsUpdated: Int,
        val lastUpdateTime: Long
    ) : RefreshResult()

    data class NetworkError(val message: String) : RefreshResult()
    data class DatabaseError(val message: String) : RefreshResult()
}

// Custom exception for network errors
class NetworkException(message: String, cause: Throwable? = null) : Exception(message, cause)