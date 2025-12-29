package com.example.yoloclskonrad

import ai.onnxruntime.*
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import androidx.compose.material3.*
import androidx.compose.foundation.shape.CircleShape


/* ---------- APP STATE ---------- */

enum class AppState {
    IDLE,
    MODEL_SELECTED,
    READY,
    RUNNING
}

/* ---------- ACTIVITY ---------- */

class MainActivity : ComponentActivity() {

    /* --- ONNX --- */
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession

    /* --- LABELS --- */
    private var labels: List<String> = emptyList()

    /* --- CAMERA --- */
    private lateinit var previewView: PreviewView
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var lastInferenceTime = 0L

    /* --- UI STATE --- */
    private var appState by mutableStateOf(AppState.IDLE)
    private var currentLabel by mutableStateOf("—")
    private var currentConfidence by mutableStateOf(0f)

    /* ---------- PERMISSIONS ---------- */

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted && appState == AppState.RUNNING) {
                startCamera()
            }
        }

    /* ---------- FILE PICKERS ---------- */

    private val pickModelLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            uri?.let {
                if (loadModelFromUri(it)) {
                    appState = AppState.MODEL_SELECTED
                }
            }
        }

    private val pickLabelsLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            uri?.let {
                if (loadLabelsFromUri(it)) {
                    appState = AppState.READY
                }
            }
        }

    /* ---------- LIFECYCLE ---------- */

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        ortEnv = OrtEnvironment.getEnvironment()

        setContent { AppUI() }
    }

    /* ---------- UI ---------- */

    @Composable
    fun AppUI() {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF0E0E11))
                .systemBarsPadding()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            /* ---------- CAMERA CARD ---------- */
            Card(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.Black
                )
            ) {
                Box(modifier = Modifier.fillMaxSize()) {

                    if (appState == AppState.RUNNING) {
                        AndroidView(
                            modifier = Modifier.fillMaxSize(),
                            factory = { context ->
                                previewView = PreviewView(context).apply {
                                    scaleType = PreviewView.ScaleType.FILL_CENTER
                                }
                                previewView
                            }
                        )

                        // Result overlay
                        Column(
                            modifier = Modifier
                                .align(Alignment.TopStart)
                                .padding(16.dp)
                                .background(
                                    color = Color.Black.copy(alpha = 0.65f),
                                    shape = RoundedCornerShape(14.dp)
                                )
                                .padding(horizontal = 16.dp, vertical = 12.dp)
                        ) {
                            Text(
                                text = currentLabel.uppercase(),
                                color = Color.White,
                                fontSize = 22.sp
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "${(currentConfidence * 100).toInt()}%",
                                color = Color.White.copy(alpha = 0.75f),
                                fontSize = 14.sp
                            )
                        }
                    } else {
                        Column(
                            modifier = Modifier
                                .align(Alignment.Center),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "Model not running",
                                color = Color.White.copy(alpha = 0.8f),
                                fontSize = 18.sp
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Select model and labels to start",
                                color = Color.White.copy(alpha = 0.5f),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }

            /* ---------- CONTROL CARD ---------- */
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF1A1A1F)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {

                    // Status row
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .background(
                                    when (appState) {
                                        AppState.IDLE -> Color.Gray
                                        AppState.MODEL_SELECTED -> Color(0xFFFFA000)
                                        AppState.READY -> Color(0xFF4CAF50)
                                        AppState.RUNNING -> Color(0xFF4CAF50)
                                    },
                                    shape = CircleShape
                                )
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = when (appState) {
                                AppState.IDLE -> "No model selected"
                                AppState.MODEL_SELECTED -> "Model selected"
                                AppState.READY -> "Ready to start"
                                AppState.RUNNING -> "Running"
                            },
                            color = Color.White.copy(alpha = 0.85f),
                            fontSize = 14.sp
                        )
                    }

                    Divider(color = Color.White.copy(alpha = 0.1f))

                    Button(
                        onClick = {
                            pickModelLauncher.launch(arrayOf("application/octet-stream"))
                        },
                        enabled = appState == AppState.IDLE,
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF2C2C34)
                        )
                    ) {
                        Text("Select model (.onnx)")
                    }

                    Button(
                        onClick = {
                            pickLabelsLauncher.launch(arrayOf("text/plain"))
                        },
                        enabled = appState == AppState.MODEL_SELECTED,
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF2C2C34)
                        )
                    ) {
                        Text("Select labels (labels.txt)")
                    }

                    Button(
                        onClick = {
                            when (appState) {
                                AppState.READY -> {
                                    appState = AppState.RUNNING
                                    requestCameraPermission.launch(android.Manifest.permission.CAMERA)
                                }
                                AppState.RUNNING -> {
                                    stopCamera()
                                    appState = AppState.READY
                                }
                                else -> {}
                            }
                        },
                        enabled = appState == AppState.READY || appState == AppState.RUNNING,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (appState == AppState.RUNNING)
                                Color(0xFFB71C1C)
                            else
                                Color(0xFF4CAF50)
                        )
                    ) {
                        Text(
                            if (appState == AppState.RUNNING) "STOP" else "START",
                            fontSize = 16.sp
                        )
                    }
                }
            }
        }
    }


    /* ---------- MODEL + LABELS ---------- */

    private fun loadModelFromUri(uri: Uri): Boolean {
        return try {
            val bytes = contentResolver.openInputStream(uri)?.readBytes() ?: return false
            if (::ortSession.isInitialized) ortSession.close()
            ortSession = ortEnv.createSession(bytes, OrtSession.SessionOptions())
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    private fun loadLabelsFromUri(uri: Uri): Boolean {
        return try {
            val text = contentResolver.openInputStream(uri)
                ?.bufferedReader()
                ?.readLines()
                ?.filter { it.isNotBlank() }
                ?: return false

            val outputSize =
                (ortSession.outputInfo.values.first().info as TensorInfo)
                    .shape[1].toInt()

            if (text.size != outputSize) {
                throw IllegalArgumentException(
                    "Labels count (${text.size}) != model classes ($outputSize)"
                )
            }

            labels = text
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    /* ---------- CAMERA ---------- */

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            val provider = cameraProvider ?: return@addListener

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                val now = System.currentTimeMillis()
                if (now - lastInferenceTime > 300) {
                    lastInferenceTime = now
                    val bitmap = imageProxyToBitmap(image)
                    val (label, conf) = classify(bitmap)
                    runOnUiThread {
                        currentLabel = label
                        currentConfidence = conf
                    }
                }
                image.close()
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        currentLabel = "—"
        currentConfidence = 0f
    }

    /* ---------- IMAGE + ML ---------- */

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val y = image.planes[0].buffer
        val u = image.planes[1].buffer
        val v = image.planes[2].buffer

        val ySize = y.remaining()
        val uSize = u.remaining()
        val vSize = v.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        y.get(nv21, 0, ySize)
        v.get(nv21, ySize, vSize)
        u.get(nv21, ySize + vSize, uSize)

        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)

        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun classify(bitmap: Bitmap): Pair<String, Float> {
        val tensor = bitmapToTensor(bitmap)
        val output = runInference(tensor)
        val idx = output.indices.maxBy { output[it] }
        return labels[idx] to output[idx]
    }

    private fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val data = FloatArray(3 * 224 * 224)

        var r = 0
        var g = 224 * 224
        var b = 2 * 224 * 224

        for (y in 0 until 224)
            for (x in 0 until 224) {
                val p = resized.getPixel(x, y)
                data[r++] = ((p shr 16) and 0xFF) / 255f
                data[g++] = ((p shr 8) and 0xFF) / 255f
                data[b++] = (p and 0xFF) / 255f
            }

        return OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(data),
            longArrayOf(1, 3, 224, 224)
        )
    }

    private fun runInference(tensor: OnnxTensor): FloatArray {
        val name = ortSession.inputNames.first()
        val output = ortSession.run(mapOf(name to tensor))[0].value as Array<FloatArray>
        return output[0]
    }
}
