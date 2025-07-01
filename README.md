# Launching rknn models on Luckfox-pico
Use this repository to run segmantation rknn model on Luckfox-pico rv1103.

### 1. Build

```
./build.sh
```

### 2. Transfer to Luckfox
```
scp -r ./install/rknn_demo_Linux/ root@172.32.0.93:/root/path/to/your/directory
```

### 3. Launch
```
./rknn_demo model/your_model.rknn model/images/your_images
```