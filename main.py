import logging
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)


def disable_gpu_usage():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    pass


if __name__ == "__main__":
    main()
