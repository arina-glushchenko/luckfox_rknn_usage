# Настройки
PROJECT_NAME="rknn_demo"
CMAKE_SYSTEM_NAME="Linux"
TARGET_SOC="."
RKNN_API_PATH="$(pwd)/include/lib/librknn_api"
CROSS_COMPILE_PREFIX="$(pwd)/include/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
INSTALL_DIR="$(pwd)/install/${PROJECT_NAME}_${CMAKE_SYSTEM_NAME}"

# Проверка зависимостей
echo "Checking dependencies..."

# Проверка наличия кросс-компилятора
if ! command -v ${CROSS_COMPILE_PREFIX}-gcc &> /dev/null; then
    echo "Error: Cross-compiler ${CROSS_COMPILE_PREFIX}-gcc not found. Please install the toolchain."
    exit 1
fi

if ! command -v ${CROSS_COMPILE_PREFIX}-g++ &> /dev/null; then
    echo "Error: Cross-compiler ${CROSS_COMPILE_PREFIX}-g++ not found. Please install the toolchain."
    exit 1
fi

# Проверка наличия CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake."
    exit 1
fi

# Проверка наличия RKNN API
if [ ! -d "${RKNN_API_PATH}" ]; then
    echo "Error: RKNN API path ${RKNN_API_PATH} does not exist. Please update RKNN_API_PATH in CMakeLists.txt and this script."
    exit 1
fi

# Проверка наличия STB заголовков
if [ ! -f "include/stb/stb_image.h" ] || [ ! -f "include/stb/stb_image_write.h" ] || [ ! -f "include/stb/stb_image_resize.h" ]; then
    echo "Error: STB headers (stb_image.h, stb_image_write.h, stb_image_resize.h) not found in include/stb."
    echo "Please download them from https://github.com/nothings/stb and place in include/stb."
    exit 1
fi

# Проверка наличия файла модели
if [ -z "$(ls model/*.rknn 2>/dev/null)" ]; then
    echo "Error: Model file not found. Please place the RKNN model in the model directory."
    exit 1
fi

# Создание директории для сборки
echo "Creating build directory..."
mkdir -p build
cd build

# Настройка CMake
echo "Configuring CMake..."
cmake \
    -DCMAKE_C_COMPILER=${CROSS_COMPILE_PREFIX}-gcc \
    -DCMAKE_CXX_COMPILER=${CROSS_COMPILE_PREFIX}-g++ \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME} \
    ..

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

# Сборка проекта
echo "Building project..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    exit 1
fi

# Установка файлов
echo "Installing files to ${INSTALL_DIR}..."
make install

if [ $? -ne 0 ]; then
    echo "Error: Installation failed."
    exit 1
fi

echo "Build and installation completed successfully!"
echo "Output files are in ${INSTALL_DIR}"