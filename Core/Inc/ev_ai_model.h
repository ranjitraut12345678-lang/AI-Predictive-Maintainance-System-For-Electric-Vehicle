/* ================================================
   ev_ai_model.h
   CDAC EV Health Monitor — STM32F407
   Updated Header
   ================================================ */

#ifndef EV_AI_MODEL_H
#define EV_AI_MODEL_H

#include <stdint.h>

/* Min-Max Scaler Parameters */
#define MIN_THROTTLE     0.0f
#define MAX_THROTTLE     100.0f

#define MIN_CURRENT      0.0f
#define MAX_CURRENT      6.0f

#define MIN_PITCH        -40.0f
#define MAX_PITCH        40.0f

#define MIN_TEMP         15.0f
#define MAX_TEMP         60.0f

/* Output Labels */
#define AI_NORMAL   0
#define AI_WARNING  1

/* Public API Prototype */
uint8_t AI_Predict(const float input[4], float *prob);

#endif /* EV_AI_MODEL_H */
