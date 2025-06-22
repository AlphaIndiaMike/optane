package com.alphaindiamike.optane.algorithms

import com.alphaindiamike.optane.model.Calculations

interface AlgorithmRepository {
    /**
     * Execute algorithm calculation and return result as string
     */
    suspend fun calculate(calculations: Calculations): String
}