package com.example.yoloclskonrad

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.background



var predictedLabel by mutableStateOf("—")
var predictedConf by mutableStateOf(0f)


class MainActivity : ComponentActivity() {
    lateinit var ortEnv: OrtEnvironment
    lateinit var ortSession: OrtSession

    lateinit var previewView: PreviewView

    var lastInferenceTime = 0L

    var currentLabel by mutableStateOf("–")
    var currentConfidence by mutableStateOf(0f)


    private val requestPermission =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                startCamera()
            }
        }


    val labels = listOf(
        "lewo",
        "prawo",
        "przód",
        "przód-lewo",
        "przód-prawo",
        "tył",
        "tył-lewo",
        "tył-prawo",
        "wnętrze"
    )


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Box(modifier = Modifier.fillMaxSize()) {

                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { context ->
                        previewView = PreviewView(context).apply {
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }
                        previewView
                    }
                )

                Text(
                    text = "${currentLabel} (${(currentConfidence * 100).toInt()}%)",
                    color = Color.White,
                    fontSize = 32.sp,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(16.dp)
                        .background(
                            Color.Black.copy(alpha = 0.6f)
                        )
                        .padding(12.dp)
                )
            }
        }



        ortEnv = OrtEnvironment.getEnvironment()

        val modelBytes = assets.open("yolo_v11_classifier.onnx").readBytes()

        ortSession = ortEnv.createSession(
            modelBytes,
            OrtSession.SessionOptions()
        )

//        val inputShape = longArrayOf(1, 3, 224, 224)
//        val inputData = FloatArray(1 * 3 * 224 * 224) { 0.0f }
//
//        val tensor = OnnxTensor.createTensor(
//            ortEnv,
//            FloatBuffer.wrap(inputData),
//            inputShape
//        )
//
//        val output = runInference(tensor)
//
//        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test)
//        val (label, conf) = classify(bitmap)
//        println("PREDICTED: $label (${conf * 100}%)")

        requestPermission.launch(android.Manifest.permission.CAMERA)

    }

    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(
                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                )
                .build()

            imageAnalysis.setAnalyzer(
                Executors.newSingleThreadExecutor()
            ) { imageProxy ->

                val now = System.currentTimeMillis()

                if (now - lastInferenceTime > 300) { // ~3 FPS
                    lastInferenceTime = now

                    val bitmap = imageProxyToBitmap(imageProxy)
                    val (label, conf) = classify(bitmap)

                    runOnUiThread {
                        currentLabel = label
                        currentConfidence = conf
                    }

                    println("PREDICTED: $label (${conf * 100}%)")
                }

                imageProxy.close()
            }


            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            90,
            out
        )

        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(
            jpegBytes,
            0,
            jpegBytes.size
        )
    }


    fun argmax(probs: FloatArray): Int {
        var maxIdx = 0
        var maxVal = probs[0]

        for (i in 1 until probs.size) {
            if (probs[i] > maxVal) {
                maxVal = probs[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    fun classify(bitmap: Bitmap): Pair<String, Float> {
        val tensor = bitmapToTensor(bitmap)
        val probs = runInference(tensor)

        val idx = argmax(probs)
        return labels[idx] to probs[idx]
    }


    fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val inputData = FloatArray(1 * 3 * 224 * 224)

        var idxR = 0
        var idxG = 224 * 224
        var idxB = 2 * 224 * 224

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)

                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                inputData[idxR++] = r
                inputData[idxG++] = g
                inputData[idxB++] = b
            }
        }

        val shape = longArrayOf(1, 3, 224, 224)

        return OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(inputData),
            shape
        )
    }


    fun runInference(inputTensor: OnnxTensor): FloatArray {
        val inputName = ortSession.inputNames.iterator().next()

        val results = ortSession.run(
            mapOf(inputName to inputTensor)
        )

        val output = results[0].value as Array<FloatArray>

        return output[0] // 9 klas
    }

}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}
