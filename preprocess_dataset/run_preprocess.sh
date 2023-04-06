# e.g.,
# bash preprocess_dataset/run_preprocess.sh /home/nas2_userF/kangyeol/Project/webtoon2022/Paint-by-Sketch/samples 7

DATA_ROOT=$1
GPU_ID=$2

IMAGE_ROOT="$DATA_ROOT/images"
SKETCH_ROOT="$DATA_ROOT/sketch"
SKETCH_BIN_ROOT="$DATA_ROOT/sketch_bin"

cd preprocess_dataset

# Run extraction
cd pidinet
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu $GPU_ID\
    --savedir $SKETCH_ROOT\
    --datadir $IMAGE_ROOT\
    --dataset Custom --evaluate trained_models/table5_pidinet.pth --evaluate-converted --eta 0.5
cd - 

# Run binarization
SKETCH_LOAD_ROOT="$SKETCH_ROOT/eval_results/imgs_epoch_019"
python binarize_sketch.py\
    --sketch_root $SKETCH_LOAD_ROOT\
    --save_root $SKETCH_BIN_ROOT

cd ..
