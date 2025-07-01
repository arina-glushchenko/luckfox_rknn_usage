cmake_minimum_required(VERSION 3.6)

project(rknn_demo)

# Настройки компилятора
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Пути к RKNN API
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/include/lib/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

# Настройка OpenCV
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/include/lib/opencv-mobile-4.10.0-luckfox-pico/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

# Директории для заголовочных файлов
include_directories(
    ${RKNN_API_PATH}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lib
    ${CMAKE_SOURCE_DIR}/include/stb
)

# Определения для stb
add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
add_definitions(-DSTB_IMAGE_WRITE_IMPLEMENTATION)
add_definitions(-DSTB_IMAGE_RESIZE_IMPLEMENTATION)

# Исходные файлы
set(SOURCES
    src/main.cc
)

# Создание исполняемого файла
add_executable(rknn_demo ${SOURCES})

# Линковка библиотек
target_link_libraries(rknn_demo
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}  # Добавляем библиотеки OpenCV
    m
)

# Установка
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_demo_${CMAKE_SYSTEM_NAME})
set(CMAKE_INSTALL_RPATH "lib")

install(TARGETS rknn_demo DESTINATION ./)
install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)

# Установка изображений (если есть)
file(GLOB IMAGE_FILES "model/*.jpg")
if(IMAGE_FILES)
    install(FILES ${IMAGE_FILES} DESTINATION ./model/)
endif()