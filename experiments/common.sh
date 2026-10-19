# Shared settings (match the paper): 50 epochs, 640x640, seed 0, optimizer=auto.
# Synthetic single-weather training uses batch 8; mixed and all AOD4 fine-tuning use batch 16.
DATA_ROOT=${DATA_ROOT:-data}
RUNS=${RUNS:-runs}
DEVICE=${DEVICE:-0}
COMMON="epochs=50 imgsz=640 seed=0 optimizer=auto device=$DEVICE"
y() { echo "$DATA_ROOT/$1/$2"; }   # y <folder> <yaml>
P1_FOG=$(y yolo_data_foggy bird_drone_foggy.yaml)
P1_SUN=$(y yolo_data_sunny bird_drone_sunny.yaml)
FOG=$(y yolo_data_foggy_17 yolo_data_foggy_17.yaml)
SUN=$(y yolo_data_sunny_17 yolo_data_sunny_17.yaml)
RAIN=$(y yolo_data_rain_17 yolo_data_rain_17.yaml)
SNOW=$(y yolo_data_snow_17 yolo_data_snow_17.yaml)
MIXED=$(y yolo_data_mixed_17 yolo_data_mixed_17.yaml)
