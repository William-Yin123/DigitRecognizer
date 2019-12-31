package com.myapps.digitrecognizer;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    PaintView paintView;
    TextView textView;
    Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        paintView = findViewById(R.id.paintView);
        textView = findViewById(R.id.textView);

        classifier = new Classifier(this);
    }

    public void check(View view) {
        Recognition recognition = classifier.classify(paintView.getScaledBitmap());

        String s = "Unable to identify digit";
        if (recognition != null) {
            s = "You wrote " + recognition.getDigit() + ". I am " + Math.round(recognition.getProb() * 100) + "% sure.";
        }

        Log.i("Paint", s);
        textView.setText(s);
    }

    public void redraw(View view) {
        paintView.redraw();
        textView.setText("Please write a digit");
    }
}
