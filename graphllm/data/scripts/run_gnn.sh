#!/bin/bash

# 图神经网络论文下载脚本
# 专门用于搜索和下载图神经网络相关的论文

echo "=========================================="
echo "arXiv图神经网络论文下载器"
echo "=========================================="

# 设置默认参数
KEYWORDS=("graph neural network" "GNN" "graph attention" "graph convolution")
MAX_RESULTS=5
SORT_BY="time"
OUTPUT_DIR="/home/chency/GraphLLM/graphllm/data/gnn_papers"
DOWNLOAD_LIMIT=5

# 显示配置信息
echo "配置信息:"
echo "  关键词: ${KEYWORDS[@]}"
echo "  最大结果数: $MAX_RESULTS"
echo "  排序方式: $SORT_BY"
echo "  输出目录: $OUTPUT_DIR"
echo "  下载数量: $DOWNLOAD_LIMIT"
echo ""

# 询问用户是否要修改参数
read -p "是否要修改参数? (y/n): " modify_params

if [[ $modify_params == "y" || $modify_params == "Y" ]]; then
    echo ""
    echo "请输入新的参数 (直接回车使用默认值):"
    
    read -p "关键词 (用空格分隔): " custom_keywords
    if [[ ! -z "$custom_keywords" ]]; then
        KEYWORDS=($custom_keywords)
    fi
    
    read -p "最大结果数 (默认: $MAX_RESULTS): " custom_max_results
    if [[ ! -z "$custom_max_results" ]]; then
        MAX_RESULTS=$custom_max_results
    fi
    
    read -p "排序方式 (time/citation/relevance, 默认: $SORT_BY): " custom_sort_by
    if [[ ! -z "$custom_sort_by" ]]; then
        SORT_BY=$custom_sort_by
    fi
    
    read -p "输出目录 (默认: $OUTPUT_DIR): " custom_output_dir
    if [[ ! -z "$custom_output_dir" ]]; then
        OUTPUT_DIR=$custom_output_dir
    fi
    
    read -p "下载数量 (默认: $DOWNLOAD_LIMIT): " custom_download_limit
    if [[ ! -z "$custom_download_limit" ]]; then
        DOWNLOAD_LIMIT=$custom_download_limit
    fi
    
    echo ""
    echo "更新后的配置:"
    echo "  关键词: ${KEYWORDS[@]}"
    echo "  最大结果数: $MAX_RESULTS"
    echo "  排序方式: $SORT_BY"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  下载数量: $DOWNLOAD_LIMIT"
    echo ""
fi

# 构建关键词字符串
keywords_str=""
for keyword in "${KEYWORDS[@]}"; do
    keywords_str="$keywords_str \"$keyword\""
done

# 询问是否显示预览
read -p "是否显示论文预览? (y/n): " show_preview
PREVIEW_FLAG=""
if [[ $show_preview == "y" || $show_preview == "Y" ]]; then
    PREVIEW_FLAG="--show_preview"
fi

# 询问是否下载PDF
read -p "是否下载PDF文件? (y/n): " download_pdfs
DOWNLOAD_FLAG=""
if [[ $download_pdfs == "y" || $download_pdfs == "Y" ]]; then
    DOWNLOAD_FLAG="--download_pdfs"
fi

# 询问是否转换为文本
CONVERT_FLAG=""
if [[ $download_pdfs == "y" || $download_pdfs == "Y" ]]; then
    read -p "是否将PDF转换为文本? (y/n): " convert_text
    if [[ $convert_text == "y" || $convert_text == "Y" ]]; then
        CONVERT_FLAG="--convert_text"
    fi
fi

echo ""
echo "开始执行..."
echo "=========================================="

# 构建并执行命令
cmd="python example_usage.py --keywords $keywords_str --max_results $MAX_RESULTS --sort_by $SORT_BY --output_dir $OUTPUT_DIR --download_limit $DOWNLOAD_LIMIT $PREVIEW_FLAG $DOWNLOAD_FLAG $CONVERT_FLAG"

echo "执行命令: $cmd"
echo ""

# 执行命令
eval $cmd

echo ""
echo "=========================================="
echo "执行完成！"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 显示输出文件信息
if [[ -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "生成的文件:"
    ls -la "$OUTPUT_DIR"
    
    if [[ -d "$OUTPUT_DIR/pdfs" ]]; then
        echo ""
        echo "PDF文件数量: $(ls -1 $OUTPUT_DIR/pdfs/*.pdf 2>/dev/null | wc -l)"
    fi
    
    if [[ -d "$OUTPUT_DIR/texts" ]]; then
        echo "文本文件数量: $(ls -1 $OUTPUT_DIR/texts/*.txt 2>/dev/null | wc -l)"
    fi
fi

echo ""
echo "脚本执行完毕！" 