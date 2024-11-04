package com.example.facerecognitiongallery

import android.net.Uri
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.facerecognitiongallery.databinding.ItemFaceBinding

class RecyclerAdapter(private val imageList: List<Uri>) : RecyclerView.Adapter<RecyclerAdapter.ViewHolder>() {
    private var clusterLabels: List<Int> = emptyList()

    class ViewHolder(val binding: ItemFaceBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemFaceBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val imageUri = imageList[position]
        holder.binding.imageView.setImageURI(imageUri)
        holder.binding.textClusterLabel.text = "Cluster: ${clusterLabels.getOrNull(position) ?: "N/A"}"
    }

    override fun getItemCount(): Int = imageList.size

    fun updateClusters(clusters: List<Int>) {
        clusterLabels = clusters
        notifyDataSetChanged()
    }
}
