import os
import tensorflow as tf
import reverb
from acme.agents.jax import builders

# 定义一个解析函数，根据需要修改特征结构
def _parse_function(proto):
    # 定义Feature结构，这里假设有整数特征和字符串特征
    feature_description = {
        'feature_name_1': tf.io.FixedLenFeature([], tf.int64),
        'feature_name_2': tf.io.FixedLenFeature([], tf.string),
        # 根据实际情况增加更多的特征
    }
    return tf.io.parse_single_example(proto, feature_description)

# 1. tensforflow 內建 module 開啟, err
def main():
    file_path = '/tmp/tmpo7fye8tu/2024-08-19T17:28:35.400953824+00:00/tables.tfrecord'
    raw_dataset = tf.data.TFRecordDataset(file_path)

    for i, raw_record in enumerate(raw_dataset):
        try:
            parsed_record = _parse_function(raw_record)
            print(f"Record {i}: {parsed_record}")
        except tf.errors.DataLossError:
            print(f"Skipping corrupted record at index {i}")

if __name__ == "__main__":
    main()