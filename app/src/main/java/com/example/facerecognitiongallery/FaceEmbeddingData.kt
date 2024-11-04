package com.example.facerecognitiongallery

import android.content.Context
import android.net.Uri
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.FileReader
import java.io.FileWriter

data class FaceEmbeddingData(
    val imageUri: String,
    val embedding: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FaceEmbeddingData
        if (imageUri != other.imageUri) return false
        return embedding.contentEquals(other.embedding)
    }

    override fun hashCode(): Int {
        var result = imageUri.hashCode()
        result = 31 * result + embedding.contentHashCode()
        return result
    }
}

//class EmbeddingStorage(private val context: Context) {
//    private val gson = Gson()
//    private val embeddingsFile = File(context.filesDir, "embeddings.json")
//
//    fun saveEmbeddings(imageUri: Uri, embedding: FloatArray) {
//        val embeddings = loadAllEmbeddings().toMutableList()
//        embeddings.add(FaceEmbeddingData(imageUri.toString(), embedding))
//
//        FileWriter(embeddingsFile).use { writer ->
//            gson.toJson(embeddings, writer)
//        }
//    }
//
//    fun loadAllEmbeddings(): List<FaceEmbeddingData> {
//        if (!embeddingsFile.exists()) return emptyList()
//
//        return try {
//            FileReader(embeddingsFile).use { reader ->
//                val type = object : TypeToken<List<FaceEmbeddingData>>() {}.type
//                gson.fromJson(reader, type)
//            }
//        } catch (e: Exception) {
//            e.printStackTrace()
//            emptyList()
//        }
//    }
//
//    fun clearEmbeddings() {
//        if (embeddingsFile.exists()) {
//            embeddingsFile.delete()
//        }
//    }
//}
