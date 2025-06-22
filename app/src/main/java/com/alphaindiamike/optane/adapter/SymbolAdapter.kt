package com.alphaindiamike.optane.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alphaindiamike.optane.R
import com.alphaindiamike.optane.model.Symbol

class SymbolAdapter(
    private val symbols: MutableList<Symbol>,
    private val onItemClick: (Symbol) -> Unit
) : RecyclerView.Adapter<SymbolAdapter.SymbolViewHolder>() {

    class SymbolViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val textSymbolName: TextView = itemView.findViewById(R.id.textSymbolName)
        val textSymbolId: TextView = itemView.findViewById(R.id.textSymbolId)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SymbolViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_symbol, parent, false)
        return SymbolViewHolder(view)
    }

    override fun onBindViewHolder(holder: SymbolViewHolder, position: Int) {
        val symbol = symbols[position]
        holder.textSymbolName.text = symbol.name
        holder.textSymbolId.text = "Last update: ${symbol.getFormattedDate(true)}"



        holder.itemView.setOnClickListener {
            onItemClick(symbol)
        }
    }

    override fun getItemCount() = symbols.size
}