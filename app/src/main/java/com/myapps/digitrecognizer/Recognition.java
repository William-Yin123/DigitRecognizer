package com.myapps.digitrecognizer;

public class Recognition {
    private final int digit;
    private final float prob;

    public Recognition(int digit, float prob) {
        this.digit = digit;
        this.prob = prob;
    }

    public int getDigit() {
        return digit;
    }

    public float getProb() {
        return prob;
    }
}
