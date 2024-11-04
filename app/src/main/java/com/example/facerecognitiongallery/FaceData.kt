package com.example.facerecognitiongallery

import android.net.Uri

data class FaceData(
    val imageUri: Uri,
    val embedding: FloatArray?
)