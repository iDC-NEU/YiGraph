#!/bin/bash

# 安装arXiv论文下载器所需的系统依赖

echo "正在安装arXiv论文下载器所需的系统依赖..."

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "检测到Debian/Ubuntu系统，使用apt-get安装..."
        sudo apt-get update
        sudo apt-get install -y poppler-utils
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        echo "检测到CentOS/RHEL系统，使用yum安装..."
        sudo yum install -y poppler-utils
    elif command -v dnf &> /dev/null; then
        # Fedora
        echo "检测到Fedora系统，使用dnf安装..."
        sudo dnf install -y poppler-utils
    else
        echo "未检测到支持的包管理器，请手动安装poppler-utils"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "检测到macOS系统，使用Homebrew安装..."
        brew install poppler
    else
        echo "请先安装Homebrew，然后运行: brew install poppler"
        exit 1
    fi
else
    echo "不支持的操作系统: $OSTYPE"
    echo "请手动安装poppler-utils"
    exit 1
fi

# 验证安装
if command -v pdftotext &> /dev/null; then
    echo "✅ pdftotext安装成功"
    pdftotext -v
else
    echo "❌ pdftotext安装失败"
    exit 1
fi

echo ""
echo "系统依赖安装完成！"
echo "现在可以运行Python脚本了。"
echo ""
echo "安装Python依赖:"
echo "pip install requests lxml PyPDF2" 