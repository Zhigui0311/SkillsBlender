#!/bin/bash
# 跳跃参数配置化训练测试脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "跳跃参数配置化 - 训练测试"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "source/standalone/workflows/rsl_rl/train.py" ]; then
    echo "错误: 请在 skillsblender 根目录运行此脚本"
    exit 1
fi

# 显示菜单
echo "请选择测试场景:"
echo "1) 场景 1: 默认配置测试（基线）"
echo "2) 场景 2: 保守配置测试（更安全）"
echo "3) 场景 3: 激进配置测试（更高效）"
echo "4) 场景 4: 自定义配置"
echo "5) 可视化测试（PLAY 模式）"
echo "6) 小规模训练测试（128 envs, 1000 iters）"
echo ""
read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "场景 1: 默认配置测试"
        echo "=========================================="
        echo "参数:"
        echo "  gap_width_threshold = 0.55"
        echo "  narrow_gap_endpoint_extension = 1.5"
        echo "  wide_gap_endpoint_extension = 2.0"
        echo "  narrow_gap_takeoff_margin = 0.4"
        echo "  wide_gap_takeoff_margin = 0.5"
        echo "  landing_margin = 0.5"
        echo "  post_jump_distance = 0.0"
        echo ""
        read -p "按 Enter 开始训练..."

        python source/standalone/workflows/rsl_rl/train.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
            --num_envs 4096 \
            --headless
        ;;

    2)
        echo ""
        echo "=========================================="
        echo "场景 2: 保守配置测试"
        echo "=========================================="
        echo "请先手动修改 go2_jump_cur_cfg.py:"
        echo ""
        echo "self.commands.path_tracking.jump_params.narrow_gap_endpoint_extension = 2.0"
        echo "self.commands.path_tracking.jump_params.wide_gap_endpoint_extension = 2.5"
        echo "self.commands.path_tracking.jump_params.narrow_gap_takeoff_margin = 0.3"
        echo "self.commands.path_tracking.jump_params.wide_gap_takeoff_margin = 0.4"
        echo "self.commands.path_tracking.jump_params.landing_margin = 0.6"
        echo "self.commands.path_tracking.jump_params.post_jump_distance = 0.5"
        echo ""
        read -p "修改完成后按 Enter 开始训练..."

        python source/standalone/workflows/rsl_rl/train.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
            --num_envs 4096 \
            --headless
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "场景 3: 激进配置测试"
        echo "=========================================="
        echo "请先手动修改 go2_jump_cur_cfg.py:"
        echo ""
        echo "self.commands.path_tracking.jump_params.narrow_gap_endpoint_extension = 1.2"
        echo "self.commands.path_tracking.jump_params.wide_gap_endpoint_extension = 1.5"
        echo "self.commands.path_tracking.jump_params.narrow_gap_takeoff_margin = 0.5"
        echo "self.commands.path_tracking.jump_params.wide_gap_takeoff_margin = 0.6"
        echo "self.commands.path_tracking.jump_params.landing_margin = 0.4"
        echo "self.commands.path_tracking.jump_params.post_jump_distance = 0.0"
        echo ""
        read -p "修改完成后按 Enter 开始训练..."

        python source/standalone/workflows/rsl_rl/train.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
            --num_envs 4096 \
            --headless
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "场景 4: 自定义配置"
        echo "=========================================="
        echo "请手动修改 go2_jump_cur_cfg.py 中的参数"
        echo ""
        read -p "修改完成后按 Enter 开始训练..."

        python source/standalone/workflows/rsl_rl/train.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
            --num_envs 4096 \
            --headless
        ;;

    5)
        echo ""
        echo "=========================================="
        echo "可视化测试（PLAY 模式）"
        echo "=========================================="
        read -p "请输入 checkpoint 路径: " checkpoint_path

        if [ -z "$checkpoint_path" ]; then
            echo "错误: checkpoint 路径不能为空"
            exit 1
        fi

        python source/standalone/workflows/rsl_rl/play.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0-PLAY \
            --num_envs 4 \
            --checkpoint "$checkpoint_path"
        ;;

    6)
        echo ""
        echo "=========================================="
        echo "小规模训练测试"
        echo "=========================================="
        echo "参数: 128 envs, 1000 iterations"
        echo ""
        read -p "按 Enter 开始训练..."

        python source/standalone/workflows/rsl_rl/train.py \
            --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
            --num_envs 128 \
            --max_iterations 1000 \
            --headless
        ;;

    *)
        echo "无效的选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir logs/"
echo ""
echo "运行可视化测试:"
echo "  python source/standalone/workflows/rsl_rl/play.py \\"
echo "    --task SkillsBlender-Path-Go2-Jump-Cur-v0-PLAY \\"
echo "    --num_envs 4 \\"
echo "    --checkpoint <path-to-checkpoint>"
