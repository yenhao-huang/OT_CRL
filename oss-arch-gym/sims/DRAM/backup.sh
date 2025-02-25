# 获取当前日期和时间，格式为 YYYYMMDDHHMMSS
date=$(date +%Y%m%d%H%M%S)

# 定义基础备份目录
base_dir=/mnt/nvme3n1/backup_$date

# 创建基础备份目录
sudo mkdir -p $base_dir

# 备份代码
sudo cp -r /home/user/Desktop/oss-arch-gym $base_dir/code

# 备份配置和性能缓冲区
sudo cp -r /home/user/Desktop/oss-arch-gym/sims/DRAM/buffer $base_dir/buffer

# 备份模型检查点
sudo cp -r /home/user/acme/checkpoint $base_dir/checkpoints

# 備份 replay buffer
sudo cp -r /tmp/replay_buffer $base_dir/replay_buffer
