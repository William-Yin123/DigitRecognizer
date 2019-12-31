package com.myapps.digitrecognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class PaintView extends View {

    private Path path = new Path();
    private Paint brush = new Paint();

    public PaintView(Context context, AttributeSet attrs) {
        super(context, attrs);

        brush.setAntiAlias(true);
        brush.setColor(Color.BLUE);
        brush.setStyle(Paint.Style.STROKE);
        brush.setStrokeJoin(Paint.Join.ROUND);
        brush.setStrokeWidth(24f);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float pointX = event.getX();
        float pointY = event.getY();

        switch(event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(pointX, pointY);
                return true;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(pointX, pointY);
                postInvalidate();
                return false;
            default:
                return false;
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawPath(path, brush);
    }

    public void redraw() {
        path.reset();
        postInvalidate();
    }

    public Bitmap getScaledBitmap() {
        setDrawingCacheEnabled(true);
        Bitmap b = Bitmap.createScaledBitmap(getDrawingCache(), 28, 28, false);
        setDrawingCacheEnabled(false);
        return b;
    }
}
