package com.example.facerecognitiongallery

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.example.facerecognitiongallery.databinding.ActivityImageDetailBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceDetector

class ImageDetailActivity : AppCompatActivity() {

    private lateinit var binding: ActivityImageDetailBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityImageDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Retrieve the image list and current position from the intent
        val imageList: ArrayList<Uri>? = intent.getParcelableArrayListExtra("image_list")
        val currentPosition = intent.getIntExtra("current_position", 0)

        // Set the image in the ImageView and detect faces
        imageList?.let {
            val imageUri = it[currentPosition]
            detectFaces(imageUri)
        }
    }

    private fun detectFaces(imageUri: Uri) {
        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(imageUri))
        val image = InputImage.fromBitmap(bitmap, 0)

        // Set up face detector options
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()

        val detector: FaceDetector = FaceDetection.getClient(options)

        detector.process(image)
            .addOnSuccessListener { faces ->
                // Draw rectangles around detected faces
                drawFaces(bitmap, faces)
            }
            .addOnFailureListener { e ->
                // Handle error
                e.printStackTrace()
            }
    }

    private fun drawFaces(bitmap: Bitmap, faces: List<Face>) {
        // Create a mutable bitmap to draw rectangles
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            strokeWidth = 8f
            style = Paint.Style.STROKE
        }

        // Draw rectangles for each detected face
        for (face in faces) {
            val boundingBox = face.boundingBox
            canvas.drawRect(boundingBox, paint)
        }

        // Set the modified bitmap to the ImageView
        binding.imageView.setImageBitmap(mutableBitmap)
    }
}
