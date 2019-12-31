package com.myapps.digitrecognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Classifier {

    public static final String MODEL_PATH = "mnist_model.tflite";

    private final Interpreter tflite;
    private final int IMAGE_SIZE = 28;
    private final int[] OUTPUT_SHAPE = {1, 10};
    private final DataType OUTPUT_DATA_TYPE = DataType.FLOAT32;
    private List<String> labels;

    public Classifier(Context context) {
        try {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, MODEL_PATH);
            Interpreter.Options tfliteOptions = new Interpreter.Options();
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            labels = new ArrayList<>();
            for (int i = 0; i <= 9; i++) {
                labels.add(String.valueOf(i));
            }
        } catch (IOException e) {
            Log.e("Digit Recognizer", "Failed to load model from " + MODEL_PATH);
            throw new RuntimeException(e);
        }
    }

    public Recognition classify(final Bitmap bitmap) {
        ByteBuffer input = convertBitmapToByteBuffer(bitmap);
        TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SHAPE, OUTPUT_DATA_TYPE);
        ByteBuffer output = outputProbabilityBuffer.getBuffer();

        tflite.run(input, output);
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().build();
        Map<String, Float> labeledProbability = new TensorLabel(
                labels,
                probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();

        Float maxProb = 0f;
        String digit = null;
        for (String d : labeledProbability.keySet()) {
            Float p = labeledProbability.get(d);
            if (p > maxProb) {
                maxProb = p;
                digit = d;
            }
        }
        if (maxProb >= 0.5) {
            return new Recognition(Integer.valueOf(digit), maxProb);
        } else {
            return null;
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        int batch_size = 1;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * batch_size * IMAGE_SIZE * IMAGE_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; i++) {
            final int value = intValues[i];
            byteBuffer.putFloat(value != -1 ? 1f : 0f);
        }

        return byteBuffer;
    }
}
