/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Production-Grade EV VCU Firmware with Embedded AI
  *                   Target      : STM32F407VGT6
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ev_ai_model.h"   /* Manual NN inference */
#include "FreeRTOS.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
  uint32_t raw_throttle;
  uint8_t  throttle_pct;
  float    motor_current_a;
  int16_t  pitch_angle;
  float    temperature_c;   // MLX90614 Live Temperature
  uint8_t  brake_status;
  uint8_t  ai_fault_status; // 0: NORMAL, 1: WARNING
  uint8_t  adc_fault;       // 0: OK, 1: ADC Read Error/Timeout
} VCU_Telemetry_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define MPU6050_ADDR         0xD0   // 0x68 << 1 (I2C Address for Gyro/Accel)
#define MLX90614_ADDR        0xB4   // 0x5A << 1 (I2C Address for IR Temp Sensor)
#define MLX90614_TOBJ1       0x07   // RAM register for Object Temperature
#define ACS712_VREF          3.3f   // STM32 ADC Reference Voltage
/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
CAN_HandleTypeDef hcan2;
I2C_HandleTypeDef hi2c1;
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart4;

/* Definitions for SafetyTask */
osThreadId_t SafetyTaskHandle;
const osThreadAttr_t SafetyTask_attributes = {
  .name = "SafetyTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
/* Definitions for ControlTask */
osThreadId_t ControlTaskHandle;
const osThreadAttr_t ControlTask_attributes = {
  .name = "ControlTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityAboveNormal,
};
/* Definitions for TelemetryTask */
osThreadId_t TelemetryTaskHandle;
const osThreadAttr_t TelemetryTask_attributes = {
  .name = "TelemetryTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};

/* USER CODE BEGIN PV */
volatile uint8_t brake_applied = 0;
char uart_buf[128];

// RTOS Inter-Process Communication Handles
osSemaphoreId_t BrakeSemaphoreHandle;
osMessageQueueId_t TelemetryQueueHandle;

// CAN Bus Variables
CAN_TxHeaderTypeDef TxHeader;
uint8_t TxData[8];
uint32_t TxMailbox;
CAN_FilterTypeDef sFilterConfig;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM2_Init(void);
static void MX_ADC1_Init(void);
static void MX_UART4_Init(void);
static void MX_CAN2_Init(void);
static void MX_I2C1_Init(void);
void StartSafetyTask(void *argument);
void StartControlTask(void *argument);
void StartTelemetryTask(void *argument);

/* USER CODE BEGIN PFP */
void MPU6050_Init(void);
int16_t MPU6050_ReadPitchAngle(void);
float MLX90614_ReadTemperature(void);
int8_t Read_ADC_Channel(uint32_t channel, uint32_t *out_val);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int8_t Read_ADC_Channel(uint32_t channel, uint32_t *out_val)
{
  ADC_ChannelConfTypeDef sConfig = {0};
  sConfig.Channel = channel;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;

  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    return -1;
  }

  if (HAL_ADC_Start(&hadc1) != HAL_OK)
  {
    return -1;
  }

  if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
  {
    *out_val = HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    return 0; // Success
  }

  HAL_ADC_Stop(&hadc1);
  return -1; // Failure
}
/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM2_Init();
  MX_ADC1_Init();
  MX_UART4_Init();
  MX_CAN2_Init();
  MX_I2C1_Init();

  /* USER CODE BEGIN 2 */
  snprintf(uart_buf, sizeof(uart_buf), "\r\n=================================\r\n");
  HAL_UART_Transmit(&huart4, (uint8_t *)uart_buf, strlen(uart_buf), 100);

  snprintf(uart_buf, sizeof(uart_buf), "   EV VCU AI/IoT Firmware v3.7   \r\n");
  HAL_UART_Transmit(&huart4, (uint8_t *)uart_buf, strlen(uart_buf), 100);

  snprintf(uart_buf, sizeof(uart_buf), "=================================\r\n\r\n");
  HAL_UART_Transmit(&huart4, (uint8_t *)uart_buf, strlen(uart_buf), 100);

  MPU6050_Init();

  sFilterConfig.FilterBank = 14;
  sFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
  sFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
  sFilterConfig.FilterIdHigh = 0x0000;
  sFilterConfig.FilterIdLow = 0x0000;
  sFilterConfig.FilterMaskIdHigh = 0x0000;
  sFilterConfig.FilterMaskIdLow = 0x0000;
  sFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
  sFilterConfig.FilterActivation = ENABLE;
  sFilterConfig.SlaveStartFilterBank = 14;

  if (HAL_CAN_ConfigFilter(&hcan2, &sFilterConfig) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_CAN_Start(&hcan2) != HAL_OK)
  {
    Error_Handler();
  }

  TxHeader.StdId = 0x103;
  TxHeader.RTR = CAN_RTR_DATA;
  TxHeader.IDE = CAN_ID_STD;
  TxHeader.DLC = 8;
  TxHeader.TransmitGlobalTime = DISABLE;

  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);

  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_1, GPIO_PIN_RESET);
  /* USER CODE END 2 */

  osKernelInitialize();

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  BrakeSemaphoreHandle = osSemaphoreNew(1, 0, NULL);
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_QUEUES */
  TelemetryQueueHandle = osMessageQueueNew(5, sizeof(VCU_Telemetry_t), NULL);
  /* USER CODE END RTOS_QUEUES */

  SafetyTaskHandle = osThreadNew(StartSafetyTask, NULL, &SafetyTask_attributes);
  ControlTaskHandle = osThreadNew(StartControlTask, NULL, &ControlTask_attributes);
  TelemetryTaskHandle = osThreadNew(StartTelemetryTask, NULL, &TelemetryTask_attributes);

  osKernelStart();

  while (1)
  {
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_ADC1_Init(void)
{
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV2;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_CAN2_Init(void)
{
  hcan2.Instance = CAN2;
  hcan2.Init.Prescaler = 2;
  hcan2.Init.Mode = CAN_MODE_NORMAL;
  hcan2.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan2.Init.TimeSeg1 = CAN_BS1_13TQ;
  hcan2.Init.TimeSeg2 = CAN_BS2_2TQ;
  hcan2.Init.TimeTriggeredMode = DISABLE;
  hcan2.Init.AutoBusOff = ENABLE;
  hcan2.Init.AutoWakeUp = DISABLE;
  hcan2.Init.AutoRetransmission = ENABLE;
  hcan2.Init.ReceiveFifoLocked = DISABLE;
  hcan2.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_I2C1_Init(void)
{
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_TIM2_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 83;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  HAL_TIM_MspPostInit(&htim2);
}

static void MX_UART4_Init(void)
{
  huart4.Instance = UART4;
  huart4.Init.BaudRate = 115200;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_TX_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0|GPIO_PIN_1, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  HAL_NVIC_SetPriority(EXTI4_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(EXTI4_IRQn);
}

/* USER CODE BEGIN 4 */
void MPU6050_Init(void)
{
  uint8_t check, data;
  if (HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, 0x75, 1, &check, 1, 100) != HAL_OK)
  {
    return;
  }
  if (check == 0x68)
  {
    data = 0x00;
    HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, 0x6B, 1, &data, 1, 100);
  }
}

int16_t MPU6050_ReadPitchAngle(void)
{
  uint8_t raw_data[6];
  int16_t accel_x, accel_y, accel_z;
  float pitch;

  if (HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, 0x3B, 1, raw_data, 6, 50) == HAL_OK)
  {
    accel_x = (int16_t)(raw_data[0] << 8 | raw_data[1]);
    accel_y = (int16_t)(raw_data[2] << 8 | raw_data[3]);
    accel_z = (int16_t)(raw_data[4] << 8 | raw_data[5]);

    float denom = sqrtf((float)accel_y * accel_y + (float)accel_z * accel_z);
    if (denom < 1.0f) denom = 1.0f;

    pitch = atan2f((float)accel_x, denom) * (180.0f / 3.14159f);
    return (int16_t)pitch;
  }
  return 0;
}

float MLX90614_ReadTemperature(void)
{
  uint8_t raw_data[3];
  int16_t temp_raw;

  if (HAL_I2C_Mem_Read(&hi2c1, MLX90614_ADDR, MLX90614_TOBJ1, 1, raw_data, 3, 100) == HAL_OK)
  {
    temp_raw = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    float temp_c = ((float)temp_raw * 0.02f) - 273.15f;
    return temp_c;
  }
  return 0.0f;
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if (GPIO_Pin == GPIO_PIN_4)
  {
    if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_4) == GPIO_PIN_RESET) {
      brake_applied = 1;
    } else {
      brake_applied = 0;
    }
    osSemaphoreRelease(BrakeSemaphoreHandle);
  }
}

void *__wrap_malloc(size_t size)
{
  return pvPortMalloc(size);
}

void __wrap_free(void *ptr)
{
  vPortFree(ptr);
}
/* USER CODE END 4 */

void StartSafetyTask(void *argument)
{
  for(;;)
  {
    if (osSemaphoreAcquire(BrakeSemaphoreHandle, osWaitForever) == osOK)
    {
      if (brake_applied)
      {
        __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
      }
    }
  }
}

void StartControlTask(void *argument)
{
  VCU_Telemetry_t tx_data;
  uint32_t adc_ch1 = 0, adc_ch2 = 0;

  float ai_input[4];
  float warning_prob = 0.0f;

  for(;;)
  {
    memset(&tx_data, 0, sizeof(VCU_Telemetry_t));

    if (!brake_applied)
    {
      int8_t res_ch1 = Read_ADC_Channel(ADC_CHANNEL_1, &adc_ch1); // PA1 Throttle
      int8_t res_ch2 = Read_ADC_Channel(ADC_CHANNEL_2, &adc_ch2); // PA2 ACS712 Current

      if (res_ch1 != 0 || res_ch2 != 0)
      {
        tx_data.adc_fault = 1;
        adc_ch1 = 0;
      }
      else
      {
        tx_data.adc_fault = 0;
      }

      // 1. Process Throttle & Motor Duty Cycle
      tx_data.raw_throttle = adc_ch1;
      uint16_t pwm_duty = (uint16_t)((adc_ch1 * 999) / 4095);
      tx_data.throttle_pct = (uint8_t)((pwm_duty * 100) / 999);
      tx_data.brake_status = 0;

      // 2. Process Current Reading
      float measured_v = ((float)adc_ch2 / 4095.0f) * 3.3f;
      float calculated_i = (measured_v - 2.5f) / 0.066f;

      if (calculated_i < 0.3f && calculated_i > -0.3f) {
        tx_data.motor_current_a = 0.0f;
      } else {
        tx_data.motor_current_a = fabsf(calculated_i);
      }

      // 3. Drive Motor PWM Output
      __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pwm_duty);

      // 4. Read Sensors via I2C Bus
      tx_data.pitch_angle = MPU6050_ReadPitchAngle();
      tx_data.temperature_c = MLX90614_ReadTemperature();

      // 5. EMBEDDED AI INFERENCE
      ai_input[0] = (float)tx_data.throttle_pct;     // Throttle (%)
      ai_input[1] = tx_data.motor_current_a;          // Current (A)
      ai_input[2] = (float)tx_data.pitch_angle;       // Pitch (deg)
      ai_input[3] = tx_data.temperature_c;           // Temperature (°C)

      tx_data.ai_fault_status = AI_Predict(ai_input, &warning_prob);
    }
    else
    {
      tx_data.raw_throttle = 0;
      tx_data.throttle_pct = 0;
      tx_data.motor_current_a = 0.0f;
      tx_data.pitch_angle = 0;
      tx_data.temperature_c = 0.0f;
      tx_data.brake_status = 1;
      tx_data.ai_fault_status = 0;
      tx_data.adc_fault = 0;
      __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
    }

    osMessageQueuePut(TelemetryQueueHandle, &tx_data, 0, 10);
    osDelay(20);
  }
}

void StartTelemetryTask(void *argument)
{
  VCU_Telemetry_t rx_data;

  for(;;)
  {
    if (osMessageQueueGet(TelemetryQueueHandle, &rx_data, NULL, osWaitForever) == osOK)
    {
      if (!rx_data.brake_status)
      {
        int current_int = (int)rx_data.motor_current_a;
        int current_dec = (int)((rx_data.motor_current_a - current_int) * 100);
        if (current_dec < 0) current_dec = -current_dec;

        int temp_int = (int)rx_data.temperature_c;
        int temp_dec = (int)((rx_data.temperature_c - temp_int) * 10);
        if (temp_dec < 0) temp_dec = -temp_dec;

        const char* ai_status = (rx_data.ai_fault_status == 0) ? "NORMAL" : "WARNING";

        snprintf(uart_buf, sizeof(uart_buf),
          "[VCU LIVE] Throttle: %3d%% | Current: %d.%02dA | Pitch: %2d deg | Temp: %d.%1d C | AI Status: %s\r\n",
          rx_data.throttle_pct, current_int, current_dec, rx_data.pitch_angle, temp_int, temp_dec, ai_status);
      }
      else
      {
        snprintf(uart_buf, sizeof(uart_buf), "[SAFETY ALERT] !!! EMERGENCY BRAKE TRIGGERED - MOTOR CUTOFF !!!\r\n");
      }
      HAL_UART_Transmit(&huart4, (uint8_t *)uart_buf, strlen(uart_buf), 50);

      // Broadcast CAN Frame
      uint16_t current_scaled = (uint16_t)(rx_data.motor_current_a * 100);
      uint8_t temp_scaled = (uint8_t)(rx_data.temperature_c);

      TxData[0] = rx_data.brake_status ? 0xFF : (rx_data.adc_fault ? 0xEE : 0x01);
      TxData[1] = rx_data.throttle_pct;
      TxData[2] = (uint8_t)(current_scaled >> 8);
      TxData[3] = (uint8_t)(current_scaled & 0xFF);
      TxData[4] = (uint8_t)(rx_data.pitch_angle >> 8);
      TxData[5] = temp_scaled;
      TxData[6] = rx_data.ai_fault_status;
      TxData[7] = rx_data.adc_fault;

      if (HAL_CAN_GetTxMailboxesFreeLevel(&hcan2) > 0)
      {
        HAL_CAN_AddTxMessage(&hcan2, &TxHeader, TxData, &TxMailbox);
      }
    }
  }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM6)
  {
    HAL_IncTick();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */
