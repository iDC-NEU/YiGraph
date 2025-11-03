#!/bin/bash

# GROBID服务启动脚本
# 用于启动GROBID服务以支持PDF结构化文本提取

echo "=========================================="
echo "GROBID服务启动脚本"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    echo "安装命令: sudo apt-get install docker.io"
    exit 1
fi

# 检查Docker服务是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker服务未运行，请启动Docker服务"
    echo "启动命令: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker环境检查通过"

# 检查GROBID容器是否已运行
if docker ps | grep -q grobid; then
    echo "✅ GROBID容器已在运行"
    echo "服务地址: http://localhost:8070"
    echo "健康检查: http://localhost:8070/api/isalive"
    exit 0
fi

# 检查GROBID镜像是否存在
if ! docker images | grep -q grobid; then
    echo "📥 拉取GROBID Docker镜像..."
    docker pull grobid/grobid:0.8.2
fi

echo "🚀 启动GROBID服务..."

# 启动GROBID容器
docker run -d \
    --gpus all \
    --name grobid \
    -p 8070:8070 \
    grobid/grobid:0.8.2

# 等待服务启动
echo "⏳ 等待GROBID服务启动..."
sleep 10

# 检查服务是否正常
for i in {1..30}; do
    if curl -s http://localhost:8070/api/isalive &> /dev/null; then
        echo "✅ GROBID服务启动成功！"
        echo "服务地址: http://localhost:8070"
        echo "健康检查: http://localhost:8070/api/isalive"
        echo ""
        echo "使用说明:"
        echo "1. 服务将在后台运行"
        echo "2. 停止服务: docker stop grobid"
        echo "3. 重启服务: docker start grobid"
        echo "4. 查看日志: docker logs grobid"
        exit 0
    fi
    echo "⏳ 等待服务启动... ($i/30)"
    sleep 2
done

echo "❌ GROBID服务启动失败"
echo "请检查Docker容器状态: docker logs grobid"
exit 1 