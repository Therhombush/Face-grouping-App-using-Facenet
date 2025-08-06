🧠 Face Detection and Grouping App
A mobile application that detects faces in user-provided images (camera or gallery), compares them with stored images, and groups similar faces using Google ML Kit and FaceNet embeddings.

🚀 Features
Take input from camera or gallery
Detect faces using Google ML Kit
Draw bounding boxes on detected faces
Generate 128D embeddings using FaceNet
Match detected face with stored faces
Group similar faces using K-Means clustering
Display matched images

🔧 Tech Stack
Feature	                         Library / Tool
Embedding	                       FaceNet (TensorFlow Lite)
Clustering	                     K-Means Algorithm
Mobile Platform	                 Android(Java/Kotlin)
Preprocessing	                   RGB normalization, Resizing

🔁 System Architecture
Input: User provides an image.
Face Detection: ML Kit detects and localizes faces.
Embedding Generation: FaceNet converts each face to a 128-dimensional vector.
Face Matching: Embeddings are compared using Euclidean Distance.
Face Grouping: Similar faces are grouped using K-Means clustering.
Output: Matched or grouped faces are shown on-screen.


<img width="100" height="200" alt="image" src="https://github.com/user-attachments/assets/8978ccba-d89c-4f38-ba29-86fc88a7d97a" />
Homescreen

<img width="100" height="200" alt="image" src="https://github.com/user-attachments/assets/635e1c3f-0a2f-4617-a972-ee2d5b592248" />
Face detection

<img width="100" height="200" alt="image" src="https://github.com/user-attachments/assets/f6948315-caf4-43a5-9d66-acadb2c83afb" />
Result


