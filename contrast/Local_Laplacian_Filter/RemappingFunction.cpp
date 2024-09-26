#include "RemappingFunction.h"

float SmoothStep(float x_min, float x_max, float x) {
    float y = (x - x_min) / (x_max - x_min);
    y = fmax(0.0, fmin(1.0, y));
    return pow(y, 2) * pow(y-2, 2);
}

float DetailRemap(float delta, float alpha, float sigma_r) {
    float fraction = delta / sigma_r;
    float polynomial = pow(fraction, alpha);
    if (alpha < 1) {
        const float kNoiseLevel = 0.01;
        float blend = SmoothStep(kNoiseLevel,
                2 * kNoiseLevel, fraction * sigma_r);
        polynomial = blend * polynomial + (1 - blend) * fraction;
    }
    return polynomial;
}

float EdgeRemap(float beta, float delta) {
    return beta * delta;
}

void Evaluate(float value, float reference, float alpha, float beta, float sigma_r, float& output) {
    float delta = abs(value - reference);
    int sign = value < reference ? -1 : 1;

    if (delta < sigma_r) {
        output = reference + sign * sigma_r * DetailRemap(delta, alpha, sigma_r);
    } else {
        output = reference + sign * (EdgeRemap(beta, delta - sigma_r) + sigma_r);
    }
}


Mat Evaluate(Mat input, float reference, float alpha, float beta, float sigma_r) {
    Mat output = Mat::zeros(input.size(), input.type());

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            Evaluate(input.at<float>(i, j), reference, alpha, beta, sigma_r, output.at<float>(i, j));
        }
    }

    return output;
}
