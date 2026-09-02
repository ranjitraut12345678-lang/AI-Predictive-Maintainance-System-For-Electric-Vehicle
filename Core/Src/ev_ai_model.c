/* ================================================
   ev_ai_model.c
   CDAC EV Health Monitor — STM32F407
   Hand-rolled forward pass with Safety Threshold Checks
   ================================================ */

#include "ev_ai_model.h"
#include <math.h>
#include <stddef.h>

/* ---- Layer 1: Dense(4 -> 16), ReLU ---- */
static const float W1[16][4] = {
    {0.5f, -0.2f, 0.1f, 0.8f}, {-0.1f, 0.4f, -0.3f, 0.2f},
    {0.3f, 0.1f, -0.5f, 0.6f}, {-0.4f, -0.2f, 0.7f, -0.1f},
    {0.2f, 0.8f, -0.1f, 0.3f}, {-0.6f, 0.1f, 0.4f, -0.5f},
    {0.1f, -0.7f, 0.2f, 0.9f}, {0.4f, 0.3f, -0.8f, -0.2f},
    {-0.3f, 0.5f, 0.6f, -0.1f}, {0.7f, -0.4f, -0.2f, 0.3f},
    {-0.2f, 0.1f, 0.5f, -0.6f}, {0.6f, 0.2f, -0.3f, 0.4f},
    {-0.5f, -0.3f, 0.1f, 0.7f}, {0.2f, -0.6f, 0.8f, -0.2f},
    {0.8f, 0.1f, -0.4f, 0.3f}, {-0.1f, 0.4f, 0.2f, -0.8f}
};
static const float B1[16] = {0.0f};

/* ---- Layer 2: Dense(16 -> 8), ReLU ---- */
static const float W2[8][16] = {
    {0.1f, 0.2f, -0.1f, 0.3f, -0.2f, 0.1f, 0.4f, -0.3f, 0.1f, -0.2f, 0.3f, -0.1f, 0.2f, 0.5f, -0.4f, 0.1f},
    {-0.2f, 0.1f, 0.4f, -0.1f, 0.3f, -0.2f, 0.1f, 0.5f, -0.3f, 0.1f, -0.4f, 0.2f, -0.1f, 0.3f, 0.2f, -0.5f},
    {0.3f, -0.4f, 0.1f, 0.2f, -0.5f, 0.4f, -0.1f, 0.2f, 0.6f, -0.1f, 0.2f, -0.3f, 0.4f, -0.2f, 0.1f, 0.3f},
    {-0.1f, 0.3f, -0.2f, 0.5f, 0.1f, -0.3f, 0.2f, -0.4f, 0.1f, 0.5f, -0.2f, 0.1f, -0.3f, 0.4f, -0.1f, 0.2f},
    {0.4f, -0.1f, 0.3f, -0.2f, 0.2f, 0.6f, -0.3f, 0.1f, -0.2f, 0.3f, 0.1f, -0.5f, 0.2f, -0.1f, 0.4f, -0.3f},
    {-0.3f, 0.2f, -0.5f, 0.1f, -0.4f, 0.2f, 0.5f, -0.1f, 0.3f, -0.4f, 0.2f, 0.1f, -0.2f, 0.3f, -0.1f, 0.5f},
    {0.2f, 0.5f, -0.1f, 0.4f, -0.3f, 0.1f, -0.2f, 0.3f, 0.4f, -0.1f, 0.3f, -0.2f, 0.1f, -0.5f, 0.2f, 0.1f},
    {-0.4f, 0.1f, 0.2f, -0.3f, 0.5f, -0.1f, 0.3f, 0.2f, -0.5f, 0.2f, -0.1f, 0.4f, -0.3f, 0.1f, 0.3f, -0.2f}
};
static const float B2[8] = {0.0f};

/* ---- Layer 3: Dense(8 -> 1), Sigmoid ---- */
static const float W3[1][8] = {
    {0.4f, -0.3f, 0.6f, -0.2f, 0.5f, -0.1f, 0.3f, 0.7f}
};
static const float B3[1] = {0.0f};

/* Activation Helpers */
static void relu_vec(float *v, int n) {
    for (int i = 0; i < n; i++) {
        if (v[i] < 0.0f) v[i] = 0.0f;
    }
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

uint8_t AI_Predict(const float input[4], float *prob)
{
    /* Input sequence: {throttle_pct, current_A, pitch_deg, temp_C} */
    float throttle = input[0];
    float current  = input[1];
    float pitch    = input[2];
    float temp     = input[3];

    /* 1. Min-Max Scaling */
    float x[4];
    x[0] = (throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE);
    x[1] = (current  - MIN_CURRENT)  / (MAX_CURRENT  - MIN_CURRENT);
    x[2] = (pitch    - MIN_PITCH)    / (MAX_PITCH    - MIN_PITCH);
    x[3] = (temp     - MIN_TEMP)     / (MAX_TEMP     - MIN_TEMP);

    for (int i = 0; i < 4; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
        if (x[i] > 1.0f) x[i] = 1.0f;
    }

    /* 2. Neural Network Forward Pass */
    float h1[16];
    for (int i = 0; i < 16; i++) {
        float acc = B1[i];
        for (int j = 0; j < 4; j++) {
            acc += W1[i][j] * x[j];
        }
        h1[i] = acc;
    }
    relu_vec(h1, 16);

    float h2[8];
    for (int i = 0; i < 8; i++) {
        float acc = B2[i];
        for (int j = 0; j < 16; j++) {
            acc += W2[i][j] * h1[j];
        }
        h2[i] = acc;
    }
    relu_vec(h2, 8);

    float acc = B3[0];
    for (int j = 0; j < 8; j++) {
        acc += W3[0][j] * h2[j];
    }
    float out_prob = sigmoid(acc);

    if (prob != NULL) {
        *prob = out_prob;
    }

    /* 3. Safety Threshold Conditions */
    if (throttle >= 70.0f || temp > 45.0f || pitch > 25.0f || pitch < -25.0f) {
        return AI_WARNING;
    }

    return AI_NORMAL;
}
