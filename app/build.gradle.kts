plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    id ("kotlin-kapt")

}

android {
    namespace = "com.example.facerecognitiongallery"
    compileSdk = 34


    buildFeatures {
        viewBinding = true
        compose = true
    }
    defaultConfig {
        applicationId = "com.example.facerecognitiongallery"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
        dataBinding = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.0"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Core Android dependencies
    implementation ("androidx.activity:activity-ktx:1.7.2")
    implementation ("androidx.fragment:fragment-ktx:1.6.0")
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.9.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    //googlemlkit
    implementation("com.google.mlkit:face-detection:16.1.6")

    //tesorflow
    implementation("org.tensorflow:tensorflow-lite:2.11.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.1.0")
    implementation ("org.tensorflow:tensorflow-lite-select-tf-ops:2.11.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.11.0")
    //implementation("libs.tensorflow.lite.gpu")
    //implementation("libs.tensorflow.lite.gpu.api")
    //implementation("libs.tensorflow.lite.support")

    implementation ("org.apache.commons:commons-math3:3.6.1")

    // RecyclerView
    implementation("androidx.recyclerview:recyclerview:1.3.1")

    // Glide for image loading
    implementation("com.github.bumptech.glide:glide:4.15.1")
    implementation(libs.firebase.crashlytics.buildtools)
    //implementation(libs.litert.gpu)
    //implementation(libs.litert)
    //implementation(libs.litert.support.api)
    kapt("com.github.bumptech.glide:compiler:4.15.1")

    implementation ("com.google.code.gson:gson:2.10.1")

    // Compose dependencies
    implementation("androidx.compose.ui:ui:1.5.0")  // Compose UI
    implementation("androidx.compose.material:material:1.5.0")  // Material Design
    implementation("androidx.compose.ui:ui-tooling-preview:1.5.0")  // Tooling
    implementation("androidx.compose.runtime:runtime:1.5.0")  // Compose Runtime

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")

    // Optional: Tooling for debugging in Compose
    debugImplementation("androidx.compose.ui:ui-tooling:1.5.0")

}