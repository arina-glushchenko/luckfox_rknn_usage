#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "stb_image.h"
#include "stb_image_write.h"

/* ==============================================
 * Вспомогательные функции
 * ============================================== */

/**
 * Получение текущего времени в микросекундах
 * @return Время в микросекундах
 */
static inline int64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * Сохранение маски сегментации в PNG файл с использованием stb_image_write
 * @param filename Имя файла для сохранения
 * @param mask Указатель на данные маски
 * @param width Ширина изображения
 * @param height Высота изображения
 */
void save_mask_as_png(const char* filename, const uint8_t* mask, int width, int height) {
    // Цвета для различных классов (в формате RGB, так как stb_image_write использует RGB)
    const uint8_t class_colors[2][3] = {
        { 255, 0, 255 }, // Фиолетовый для класса 0
        { 0, 0, 0 }      // Черный для класса 1
    };

    // Создание цветного изображения
    uint8_t* rgb_image = (uint8_t*)malloc(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        uint8_t class_id = mask[i];
        rgb_image[i * 3 + 0] = class_colors[class_id][0]; // R
        rgb_image[i * 3 + 1] = class_colors[class_id][1]; // G
        rgb_image[i * 3 + 2] = class_colors[class_id][2]; // B
    }

    // Сохранение изображения с помощью stb_image_write
    if (stbi_write_png(filename, width, height, 3, rgb_image, width * 3)) {
        printf("Saved colored mask: %s\n", filename);
    } else {
        printf("Failed to save PNG: %s\n", filename);
    }

    free(rgb_image);
}

/* ==============================================
 * Основная функция
 * ============================================== */

int main(int argc, char **argv) {
    // Проверка аргументов командной строки
    if (argc < 3) {
        printf("Usage: %s <model.rknn> <input.jpg>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int64_t start, end;

    /* ------------------------------
     * Инициализация модели
     * ------------------------------ */
    start = get_time_us();
    
    // Загрузка модели
    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    rewind(fp);
    
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    // Инициализация контекста RKNN
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret != RKNN_SUCC) {
        printf("rknn_init failed: %d\n", ret);
        return -1;
    }

    /* ------------------------------
     * Подготовка входных данных
     * ------------------------------ */
    
    // Получение атрибутов входного тензора
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    int input_h = input_attr.dims[1];
    int input_w = input_attr.dims[2];
    int input_c = input_attr.dims[3];
    printf("Model input: %d x %d x %d\n", input_h, input_w, input_c);

    // Загрузка входного изображения с помощью stb_image
    int img_width, img_height, img_channels;
    unsigned char *img_data = stbi_load(image_path, &img_width, &img_height, &img_channels, input_c);
    if (!img_data) {
        printf("Failed to load image: %s\n", image_path);
        rknn_destroy(ctx);
        return -1;
    }

    // Создание cv::Mat из загруженного изображения
    cv::Mat input_img(img_height, img_width, CV_8UC3, img_data);
    if (input_img.empty()) {
        printf("Failed to create cv::Mat from image: %s\n", image_path);
        stbi_image_free(img_data);
        rknn_destroy(ctx);
        return -1;
    }

    // Изменение размера изображения
    cv::Mat resized_img;
    cv::resize(input_img, resized_img, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    printf("Loaded image %s (%d x %d x %d), resized to %d x %d x %d\n", 
           image_path, img_width, img_height, img_channels, 
           input_w, input_h, input_c);

    // Освобождение данных изображения
    stbi_image_free(img_data);

    // Проверка соответствия количества каналов
    if (resized_img.channels() != input_c) {
        printf("Image channels (%d) do not match model input channels (%d)\n", resized_img.channels(), input_c);
        rknn_destroy(ctx);
        return -1;
    }

    // Выделение памяти и привязка входного тензора
    rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_h * input_w * input_c);
    memcpy(input_mem->virt_addr, resized_img.data, input_h * input_w * input_c);
    ret = rknn_set_io_mem(ctx, input_mem, &input_attr);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem failed: %d\n", ret);
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy(ctx);
        return -1;
    }

    /* ------------------------------
     * Подготовка выхода модели
     * ------------------------------ */
    
    // Получение атрибутов выходного тензора
    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    int out_h = output_attr.dims[1];
    int out_w = output_attr.dims[2];
    int out_c = output_attr.dims[3];
    printf("Model output: %d x %d x %d\n", out_h, out_w, out_c);

    // Выделение памяти и привязка выходного тензора
    rknn_tensor_mem* output_mem = rknn_create_mem(ctx, out_h * out_w * out_c);
    ret = rknn_set_io_mem(ctx, output_mem, &output_attr);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem (output) failed: %d\n", ret);
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy_mem(ctx, output_mem);
        rknn_destroy(ctx);
        return -1;
    }
    
    end = get_time_us();
    float preproc_time = (end - start) / 1000.0f;
    
    /* ------------------------------
     * Выполнение вывода модели
     * ------------------------------ */
    printf("Running inference...\n");
    start = get_time_us();
    ret = rknn_run(ctx, nullptr);
    end = get_time_us();
    if (ret != RKNN_SUCC) {
        printf("rknn_run failed: %d\n", ret);
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy_mem(ctx, output_mem);
        rknn_destroy(ctx);
        return -1;
    }
    float infer_time = (end - start) / 1000.0f;

    /* ------------------------------
     * Постобработка результатов
     * ------------------------------ */
    start = get_time_us();
    uint8_t* output_data = (uint8_t*)output_mem->virt_addr;

    // Постобработка - argmax по каналам
    int out_size = out_h * out_w;
    uint8_t* seg_mask = (uint8_t*)malloc(out_size);

    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            int base = (h * out_w + w) * out_c;
            int max_class = 0;
            uint8_t max_val = output_data[base];
            
            for (int c = 1; c < out_c; ++c) {
                uint8_t val = output_data[base + c];
                if (val > max_val) {
                    max_val = val;
                    max_class = c;
                }
            }
            seg_mask[h * out_w + w] = max_class;
        }
    }

    // Сохранение результата
    save_mask_as_png("seg_mask.png", seg_mask, out_w, out_h);
    end = get_time_us();

    /* ------------------------------
     * Завершение работы
     * ------------------------------ */
    float postproc_time = (end - start) / 1000.0f;
    printf("Preprocess time: %.2f ms\n", preproc_time);
    printf("Inference time: %.2f ms\n", infer_time);
    printf("Postprocess time: %.2f ms\n", postproc_time);
    
    // Освобождение ресурсов
    free(seg_mask);
    rknn_destroy_mem(ctx, input_mem);
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);

    return 0;
}