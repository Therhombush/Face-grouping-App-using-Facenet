package com.example.facerecognitiongallery

import ImageAdapter
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.util.ArrayList

class MatchingImagesActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ImageAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_matching_images)

        // Retrieve the matched images from the intent
        val matchedImages = intent.getParcelableArrayListExtra<Uri>("matched_images") ?: ArrayList()

        // Show a message if no images are found
        if (matchedImages.isEmpty()) {
            Toast.makeText(this, "No matches found", Toast.LENGTH_SHORT).show()
            finish() // Close the activity if no matches found
            return
        }

        // Set up the RecyclerView to display the matched images
        recyclerView = findViewById(R.id.recyclerView)
        recyclerView.layoutManager = GridLayoutManager(this, 3) // 2 columns for grid layout

        // Initialize the adapter with the matched images
        adapter = ImageAdapter(
            imageUriList = matchedImages,
            imageBitmapList = ArrayList(), // Empty bitmap list since we're using URIs
            onImageClick = { imageUri ->
                val intent = Intent(this, ImageDetailActivity::class.java)
                intent.putParcelableArrayListExtra("image_list", matchedImages)
                intent.putExtra("current_position", matchedImages.indexOf(imageUri))
                startActivity(intent)
            }
        )

        // Set the adapter to RecyclerView
        recyclerView.adapter = adapter
    }
}