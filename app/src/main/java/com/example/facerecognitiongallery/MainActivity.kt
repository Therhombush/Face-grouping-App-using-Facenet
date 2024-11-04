package com.example.facerecognitiongallery

import ImageAdapter
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import com.example.facerecognitiongallery.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.atan2
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var imageAdapter: ImageAdapter
    private var imageList: ArrayList<Uri> = arrayListOf()
    private var embeddingList: ArrayList<FloatArray> = arrayListOf()  // Store embeddings of saved images
    private lateinit var interpreter: Interpreter  // TensorFlow Lite interpreter for facenet.tflite

    private val ADD_IMAGES_REQUEST_CODE = 1003
    private val CAMERA_REQUEST_CODE = 1004
    private val SELECT_INPUT_PHOTO_REQUEST_CODE = 1005
    //private val COSINE_SIMILARITY_THRESHOLD = 0.7f  // Threshold for face matching
    //private val MAX_ROTATION_DEGREES = 15f
    private val MIN_FACE_SIZE = 0.2f
    //private val BRIGHTNESS_STEPS = 3
    //private val BRIGHTNESS_DELTA = 0.2f
    private val FACE_MATCH_MIN_CONFIDENCE = 0.98f  // High confidence threshold for face detection
    private val COSINE_SIMILARITY_THRESHOLD = 0.75f // Slightly increased threshold
    private val BRIGHTNESS_STEPS = 5  // Increased from 3
    private val BRIGHTNESS_DELTA = 0.15f  // Reduced from 0.2
    private val MAX_ROTATION_DEGREES = 10f  // Reduced from 15
    private val FACE_PADDING_PERCENT = 0.2f // 20% padding around detected face
    private val faceDataList = mutableListOf<FaceData>()


    private data class FaceQuality(
        val sharpness: Float,
        val brightness: Float,
        val symmetry: Float
    )



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize TensorFlow Lite interpreter
        initTFLiteInterpreter()

        // Setup RecyclerView with a grid layout
        binding.recyclerView.layoutManager = GridLayoutManager(this, 3)
        imageAdapter = ImageAdapter(
            imageUriList = imageList, // Pass your list of URIs
            imageBitmapList = ArrayList(), // Pass an empty Bitmap list or your actual Bitmap list if you have any
            onImageClick = { imageUri ->
                val intent = Intent(this, ImageDetailActivity::class.java)
                intent.putParcelableArrayListExtra("image_list", imageList)
                intent.putExtra("current_position", imageList.indexOf(imageUri))
                startActivity(intent)
            }
        )
        binding.recyclerView.adapter = imageAdapter

        // Set click listener for the "Select Photo" button for face matching
        binding.selectPhotoButton.setOnClickListener {
            selectInputPhotoForProcessing()
        }

        // Set click listener for the "Add Images" button
        binding.addImagesButton.setOnClickListener {
            addImagesFromGallery()
        }

        // Set click listener for the "Open Camera" button
        binding.openCameraButton.setOnClickListener {
            openCamera()
        }

        // Load saved images and embeddings when app starts
        loadImagesFromStorage()

        loadEmbeddingsFromFile()

    }

    private fun alignFace(bitmap: Bitmap, face: Face): Bitmap {
        val rotationMatrix = Matrix()
        rotationMatrix.postRotate(-face.headEulerAngleZ) // Correct for roll (tilt)

        return Bitmap.createBitmap(
            bitmap,
            face.boundingBox.left,
            face.boundingBox.top,
            face.boundingBox.width(),
            face.boundingBox.height(),
            rotationMatrix,
            true
        )
    }



    private fun logModelDetails() {
        if (::interpreter.isInitialized) {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            Log.d("TFLite", "Input tensor shape: [${inputTensor.shape().joinToString()}]")
            Log.d("TFLite", "Input tensor bytes: ${inputTensor.numBytes()}")
            Log.d("TFLite", "Output tensor shape: [${outputTensor.shape().joinToString()}]")
        }
    }
    private fun createFaceDetectorOptions(): FaceDetectorOptions {
        return FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(MIN_FACE_SIZE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
    }

    private fun initTFLiteInterpreter() {
        try {
            val modelFile = File(filesDir, "facenet.tflite")
            if (!modelFile.exists()) {
                assets.open("facenet.tflite").use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.d("TFLite", "Model file copied to: ${modelFile.absolutePath}")
            }

            // Create interpreter
            val options = Interpreter.Options()
            interpreter = Interpreter(modelFile, options)

            // Log model details for debugging
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            Log.d("TFLite", "Input tensor shape: [${inputTensor.shape().joinToString()}]")
            Log.d("TFLite", "Output tensor shape: [${outputTensor.shape().joinToString()}]")

        } catch (e: Exception) {
            Log.e("TFLite", "Error in initialization: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun getFaceEmbedding(bitmap: Bitmap): FloatArray? {
        if (!::interpreter.isInitialized) {
            Log.e("TFLite", "Interpreter is not initialized.")
            return null
        }

        return try {
            // Resize to 160x160 as expected by FaceNet
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true)

            // Preprocess the image
            val inputBuffer = preprocessImageForModel(resizedBitmap)

            // Create output buffer for 128-dimensional embedding
            val outputBuffer = ByteBuffer.allocateDirect(1 * 128 * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // Run inference
            interpreter.run(inputBuffer, outputBuffer)

            // Convert output buffer to float array
            outputBuffer.rewind()
            val embeddings = FloatArray(128)
            for (i in 0 until 128) {
                embeddings[i] = outputBuffer.float
            }

            embeddings
        } catch (e: Exception) {
            Log.e("TFLite", "Error extracting face embeddings: ${e.message}")
            e.printStackTrace()
            null
        }
    }



    // Helper method to preprocess a bitmap image for the TensorFlow Lite model
    private fun preprocessImageForModel(bitmap: Bitmap): ByteBuffer {
        val modelInputSize = 160
        val channels = 3
        val numPixels = modelInputSize * modelInputSize

        // Allocate byte buffer with 4 bytes per channel per pixel
        val imgData = ByteBuffer.allocateDirect(1 * modelInputSize * modelInputSize * channels * 4)
        imgData.order(ByteOrder.nativeOrder())

        val intValues = IntArray(numPixels)
        bitmap.getPixels(intValues, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)

        // Convert the image to floating point
        var pixel = 0
        for (i in 0 until modelInputSize) {
            for (j in 0 until modelInputSize) {
                val value = intValues[pixel++]

                // Normalize to [-1,1] and write to ByteBuffer
                imgData.putFloat(((value shr 16 and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat(((value shr 8 and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat((value and (0xFF - 127.5f).toInt()) / 127.5f)
            }
        }

        imgData.rewind()
        return imgData
    }

    // Handle the result from the gallery and camera
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                ADD_IMAGES_REQUEST_CODE -> {
                    if (data?.clipData != null) {
                        val count = data.clipData!!.itemCount
                        for (i in 0 until count) {
                            val imageUri = data.clipData!!.getItemAt(i).uri
                            saveImageToLocalStorage(imageUri)
                            extractFaceEmbedding(imageUri)
                        }
                    } else if (data?.data != null) {
                        val imageUri = data.data
                        imageUri?.let {
                            saveImageToLocalStorage(it)
                            extractFaceEmbedding(it)
                        }
                    }
                    imageAdapter.notifyDataSetChanged()
                }

                CAMERA_REQUEST_CODE -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    detectAndMatchFaceFromBitmap(imageBitmap)
                    //val imageUri = saveCameraImage(imageBitmap)
                    //imageUri?.let {
                    //    imageList.add(it)
                    //    imageAdapter.notifyDataSetChanged()
                    //    extractFaceEmbedding(it)
                    //}
                }

                SELECT_INPUT_PHOTO_REQUEST_CODE -> {
                    val imageUri = data?.data
                    if (imageUri != null) {
                        detectAndMatchFace(imageUri)
                    }
                }
            }
        }
    }


    private fun detectAndMatchFaceFromBitmap(bitmap: Bitmap) {
        val faceDetectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()

        val faceDetector = FaceDetection.getClient(faceDetectorOptions)
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    val face = faces[0]
                    val croppedFaceBitmap = cropToFace(bitmap, face)
                    val inputEmbedding = getFaceEmbedding(croppedFaceBitmap)

                    // Perform face matching
                    val matchedImages = inputEmbedding?.let { findMatchingImages(it) }

                    // Display matched images
                    if (matchedImages != null) {
                        if (matchedImages.isNotEmpty()) {
                            val intent = Intent(this@MainActivity, MatchingImagesActivity::class.java)
                            intent.putParcelableArrayListExtra("matched_images", matchedImages)
                            startActivity(intent)
                        } else {
                            Toast.makeText(this@MainActivity, "No matches found", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
            .addOnFailureListener { e ->
                Log.e("Face Detection", "Failed: ${e.message}")
            }
    }



    // Extract face embedding using the new FaceNet model
    private fun extractFaceEmbedding(imageUri: Uri) {
        CoroutineScope(Dispatchers.IO).launch {
            val inputStream = contentResolver.openInputStream(imageUri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (bitmap == null) {
                Log.e("FaceEmbedding", "Bitmap is null. Cannot create InputImage.")
                return@launch
            }

            val faceDetectorOptions = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .build()

            val faceDetector = FaceDetection.getClient(faceDetectorOptions)
            val inputImage = InputImage.fromBitmap(bitmap, 0)

            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    Log.i("FaceEmbedding", "Faces detected: ${faces.size}") // Log the number of faces detected
                    if (faces.isNotEmpty()) {
                        val face = faces[0]
                        // Use the cropped face to extract embeddings
                        val croppedFaceBitmap = cropToFace(bitmap, face)
                        val embedding = getFaceEmbedding(croppedFaceBitmap)
                        if (embedding != null) {
                            embeddingList.add(embedding) // Store the embedding for later use
                            saveEmbeddingsToFile()  // Save embeddings after extraction
                        } else {
                            Log.e("FaceEmbedding", "Failed to extract embedding for this face.")
                        }
                    } else {
                        Log.i("FaceEmbedding", "No faces detected.")
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("FaceDetection", "Failed to detect faces: ${e.message}")
                }
        }
    }


    // Create face variations with embedding
    private fun createFaceVariations(embedding: FloatArray): List<FloatArray> {
        val variations = mutableListOf<FloatArray>()
        variations.add(embedding)  // Add original embedding

        // Add slightly perturbed versions
        for (i in 1..3) {
            val perturbedEmbedding = embedding.clone()
            for (j in perturbedEmbedding.indices) {
                perturbedEmbedding[j] += (Random.nextFloat() - 0.5f) * 0.01f
            }
            variations.add(perturbedEmbedding)
        }

        return variations
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    // Utility function to adjust brightness
    private fun adjustBrightness(source: Bitmap, factor: Float): Bitmap {
        val cm = ColorMatrix()
        cm.setScale(factor, factor, factor, 1f)

        val resultBitmap = Bitmap.createBitmap(source.width, source.height, source.config)
        val canvas = Canvas(resultBitmap)
        val paint = Paint()
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(source, 0f, 0f, paint)

        return resultBitmap
    }


    // Function to crop a face from the original bitmap based on the detected face boundaries
    private fun cropToFace(bitmap: Bitmap, face: Face): Bitmap {
        // Get the face's bounding box
        val bounds = face.boundingBox

        // Calculate the coordinates for cropping
        val x = bounds.left.coerceAtLeast(0) // Ensure it doesn't go out of bounds
        val y = bounds.top.coerceAtLeast(0)
        val width = bounds.width().coerceAtMost(bitmap.width - x) // Ensure it doesn't exceed the image's width
        val height = bounds.height().coerceAtMost(bitmap.height - y)

        // Crop the face from the bitmap
        return Bitmap.createBitmap(bitmap, x, y, width, height)
    }

    // Method to detect faces and match with saved embeddings
    private fun detectAndMatchFace(imageUri: Uri) {
        if (!::interpreter.isInitialized) {
            Log.e("FaceMatching", "Interpreter not initialized, cannot process image")
            Toast.makeText(this, "Face recognition model not ready", Toast.LENGTH_SHORT).show()
            return
        }
        CoroutineScope(Dispatchers.IO).launch {
            val inputStream = contentResolver.openInputStream(imageUri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (bitmap == null) {
                Log.e("FaceMatching", "Bitmap is null. Cannot create InputImage.")
                return@launch
            }

            val faceDetectorOptions = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .build()

            val faceDetector = FaceDetection.getClient(faceDetectorOptions)
            val inputImage = InputImage.fromBitmap(bitmap, 0)

            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        val face = faces[0]
                        val croppedFaceBitmap = cropToFace(bitmap, face)
                        val inputEmbedding = getFaceEmbedding(croppedFaceBitmap)

                        // Perform face matching
                        val matchedImages = inputEmbedding?.let { findMatchingImages(it) }

                        // Display matched images
                        if (matchedImages != null) {
                            if (matchedImages.isNotEmpty()) {
                                val intent = Intent(this@MainActivity, MatchingImagesActivity::class.java)
                                intent.putParcelableArrayListExtra("matched_images", matchedImages)
                                startActivity(intent)
                            } else {
                                Toast.makeText(this@MainActivity, "No matches found", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("Face Detection", "Failed: ${e.message}")
                }
        }
    }


//    private fun addNewFace(imageUri: Uri, embedding: FloatArray?) {
//        synchronized(faceDataList) {
//            faceDataList.add(FaceData(imageUri, embedding))
//        }
//    }

    // Compare input image embedding with saved embeddings and find matching images
    private fun findMatchingImages(inputEmbedding: FloatArray): ArrayList<Uri> {
        val matchedImages = arrayListOf<Uri>()

        for (i in embeddingList.indices) {
            val similarity = calculateCosineSimilarity(inputEmbedding, embeddingList[i])
            if (similarity > COSINE_SIMILARITY_THRESHOLD) {
                matchedImages.add(imageList[i])
            }
        }

        return matchedImages
    }

    // Calculate cosine similarity between two embeddings
    private fun calculateCosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != 128 || embedding2.size != 128) {
            Log.e("Similarity", "Invalid embedding dimensions")
            return 0f
        }

        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            normA += embedding1[i].pow(2)
            normB += embedding2[i].pow(2)
        }

        return if (normA == 0f || normB == 0f) {
            0f
        } else {
            dotProduct / (sqrt(normA) * sqrt(normB))
        }
    }



    // Add image from gallery to the local storage
    private fun addImagesFromGallery() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
        intent.type = "image/*"
        startActivityForResult(intent, ADD_IMAGES_REQUEST_CODE)
    }

    // Open the camera and take a picture
    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST_CODE)
    }

    // Select a photo for face matching
    private fun selectInputPhotoForProcessing() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        startActivityForResult(intent, SELECT_INPUT_PHOTO_REQUEST_CODE)
    }

    // Load saved images from local storage
    private fun loadImagesFromStorage() {
        val imageDir = File(filesDir, "images")
        if (imageDir.exists()) {
            imageDir.listFiles()?.forEach { file ->
                imageList.add(Uri.fromFile(file))
            }
            imageAdapter.notifyDataSetChanged()
        }
    }

    // Save image to local storage from Uri
    private fun saveImageToLocalStorage(imageUri: Uri) {
        val imageDir = File(filesDir, "images")
        if (!imageDir.exists()) {
            imageDir.mkdir()
        }

        val inputStream = contentResolver.openInputStream(imageUri)
        val file = File(imageDir, "${System.currentTimeMillis()}.jpg")
        val outputStream = FileOutputStream(file)

        inputStream?.use { input ->
            outputStream.use { output ->
                input.copyTo(output)
            }
        }

        imageList.add(Uri.fromFile(file))
    }

    // Save image from camera capture to local storage
    private fun saveCameraImage(bitmap: Bitmap): Uri? {
        val imageDir = File(filesDir, "images")
        if (!imageDir.exists()) {
            imageDir.mkdir()
        }

        val file = File(imageDir, "${System.currentTimeMillis()}.jpg")
        val outputStream = FileOutputStream(file)

        outputStream.use {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
        }

        return Uri.fromFile(file)
    }

    private fun saveEmbeddingsToFile() {
        try {
            val file = File(filesDir, "embeddings.txt")
            FileOutputStream(file).use { fos ->
                embeddingList.forEach { embedding ->
                    fos.write(embedding.joinToString(",").toByteArray())
                    fos.write("\n".toByteArray()) // New line for each embedding
                }
            }
        } catch (e: IOException) {
            Log.e("Embeddings", "Error saving embeddings: ${e.message}")
        }
    }

    private fun loadEmbeddingsFromFile() {
        val file = File(filesDir, "embeddings.txt")
        if (!file.exists()) return

        try {
            file.readLines().forEach { line ->
                val embedding = line.split(",").map { it.toFloat() }.toFloatArray()
                embeddingList.add(embedding)
            }
            Log.d("Embeddings", "Loaded ${embeddingList.size} embeddings from file")
        } catch (e: IOException) {
            Log.e("Embeddings", "Error loading embeddings: ${e.message}")
        }
    }
    


}
