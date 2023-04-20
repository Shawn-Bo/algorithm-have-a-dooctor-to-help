project_root_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd);  # 脚本文件的所在目录的绝对路径
export PYTHONPATH=$PYTHONPATH:$project_root_dir
echo "PYTHONPATH ADDED: "$project_root_dir