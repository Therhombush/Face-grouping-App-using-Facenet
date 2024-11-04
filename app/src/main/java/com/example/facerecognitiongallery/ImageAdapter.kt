import android.graphics.Bitmap
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.example.facerecognitiongallery.R

class ImageAdapter(
    private var imageUriList: ArrayList<Uri>? = ArrayList(), // Default to an empty list
    private var imageBitmapList: ArrayList<Bitmap> = ArrayList(), // Default to an empty list
    private var onImageClick: (Any) -> Unit // Lambda for handling click events
) : RecyclerView.Adapter<ImageAdapter.ImageViewHolder>() {

    private var selectionMode = false
    private var selectedPositions = ArrayList<Int>()

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_image, parent, false)
        return ImageViewHolder(view)
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        // Adjust alpha based on selection mode
        holder.itemView.alpha = if (selectionMode && selectedPositions.contains(position)) 0.5f else 1f

        // Check if imageUriList is not empty and has the current position
        if (imageUriList!!.isNotEmpty() && position < imageUriList!!.size) {
            val imageUri = imageUriList!![position]
            // Use Glide to load images from URI
            Glide.with(holder.itemView.context)
                .load(imageUri)
                .into(holder.imageView)

            holder.itemView.setOnClickListener {
                if (selectionMode) {
                    toggleSelection(position)
                } else {
                    onImageClick(imageUri) // Pass Uri
                }
            }
        }

        // Check if imageBitmapList is not empty and has the current position
        if (imageBitmapList.isNotEmpty() && position < imageBitmapList.size) {
            val imageBitmap = imageBitmapList[position]
            // Load images from Bitmap
            holder.imageView.setImageBitmap(imageBitmap)

            holder.itemView.setOnClickListener {
                if (selectionMode) {
                    toggleSelection(position)
                } else {
                    onImageClick(imageBitmap) // Pass Bitmap
                }
            }
        }

        // Handle long click for entering selection mode
        holder.itemView.setOnLongClickListener {
            if (!selectionMode) {
                setSelectionMode(true)
                toggleSelection(position)
            }
            true
        }
    }

    override fun getItemCount(): Int {
        return imageUriList?.size ?: imageBitmapList?.size ?: 0
    }

    // Toggle selection mode
    private fun toggleSelection(position: Int) {
        if (selectedPositions.contains(position)) {
            selectedPositions.remove(position)
        } else {
            selectedPositions.add(position)
        }
        notifyItemChanged(position)
    }

    // Enable or disable selection mode
    fun setSelectionMode(enabled: Boolean) {
        selectionMode = enabled
        selectedPositions.clear()
        notifyDataSetChanged()
    }

    // Remove selected images from URI list
    fun removeSelectedImages() {
        selectedPositions.sortDescending()
        imageUriList?.let {
            for (position in selectedPositions) {
                it.removeAt(position)
            }
        }
        selectedPositions.clear()
        setSelectionMode(false)
        notifyDataSetChanged()
    }

    // Update the adapter to display matched images (Bitmaps)
    fun updateBitmapList(newImageList: List<Bitmap>) {
        imageUriList?.clear() // Clear the URI list since we're using Bitmap
        imageBitmapList.clear() // Clear the existing Bitmap list
        imageBitmapList.addAll(newImageList) // Add new Bitmaps
        notifyDataSetChanged() // Refresh the RecyclerView
    }

    // Set click listener
    fun setOnItemClickListener(listener: (Any) -> Unit) {
        onImageClick = listener
    }

    // ViewHolder class for holding image views
    class ImageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.imageView)
    }
}
