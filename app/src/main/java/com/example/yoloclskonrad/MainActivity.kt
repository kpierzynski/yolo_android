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
                .background(Color.Black)
                .systemBarsPadding()
        ) {

            /* --- CAMERA PREVIEW --- */
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
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

                    Column(
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 16.dp)
                            .background(
                                Color.Black.copy(alpha = 0.65f),
                                RoundedCornerShape(12.dp)
                            )
                            .padding(horizontal = 16.dp, vertical = 10.dp)
                    ) {
                        Text(currentLabel, color = Color.White, fontSize = 26.sp)
                        Text(
                            "${(currentConfidence * 100).toInt()}%",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 16.sp
                        )
                    }
                } else {
                    Text(
                        text = "Wybierz model i labelki",
                        color = Color.White.copy(alpha = 0.6f),
                        modifier = Modifier.align(Alignment.Center),
                        fontSize = 18.sp
                    )
                }
            }

            /* --- CONTROLS --- */
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Color(0xFF111111),
                        RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)
                    )
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                Button(
                    onClick = {
                        pickModelLauncher.launch(arrayOf("application/octet-stream"))
                    },
                    enabled = appState == AppState.IDLE
                ) {
                    Text("Wybierz model (.onnx)")
                }

                Button(
                    onClick = {
                        pickLabelsLauncher.launch(arrayOf("text/plain"))
                    },
                    enabled = appState == AppState.MODEL_SELECTED
                ) {
                    Text("Wybierz labelki (labels.txt)")
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
                    enabled = appState == AppState.READY || appState == AppState.RUNNING
                ) {
                    Text(if (appState == AppState.RUNNING) "STOP" else "START")
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
