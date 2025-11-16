document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
        startOnLoad: false,
        theme: document.documentElement.classList.contains('dark') ? 'dark' : 'light', 
        flowchart: {
        curve: '' // 连接线样式
        }
    });

    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const modelRadios = document.querySelectorAll('input[name="language-model"]');
    
    let currentThinkingBlock = null;
    let currentResultBlock = null;
    let currentMessageId = null;
    let isProcessing = false;
    let currentConversationContainer = null;
    let currentDagContainer = null; 
    let isDagReviewMode = false; 
    let currentDagId = null; 
    let isDagModeEnabled = false;
    // 初始化变量存储当前选择
    let selectedModel = 'GPT 4'; // 默认模型
    let selectedDataset = null; // 默认无数据集

    // 检查本地存储中的DAG模式设置
    if (localStorage.getItem('dagModeEnabled') !== null) {
        isDagModeEnabled = localStorage.getItem('dagModeEnabled') === 'true';
        document.getElementById('dag-mode-toggle').checked = isDagModeEnabled;
    }

    // 处理DAG模式切换
    document.getElementById('dag-mode-toggle').addEventListener('change', function(e) {
        isDagModeEnabled = e.target.checked;
        localStorage.setItem('dagModeEnabled', isDagModeEnabled);
    });

    // 加载模型列表
    async function loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            if (data.success) {
                populateModelDropdown(data.data);
            }
        } catch (error) {
            console.error('加载模型列表失败:', error);
        }
    }

    // 加载数据集列表
    async function loadDatasets() {
        try {
            const response = await fetch('/api/knowledge_bases');
            const data = await response.json();
            if (data.success) {
                populateDatasetDropdown(data.data);
            }
        } catch (error) {
            console.error('加载数据集列表失败:', error);
        }
    }

    // 填充模型下拉框
    function populateModelDropdown(models) {
        const dropdown = document.getElementById('model-dropdown');
        dropdown.innerHTML = '';
        
        models.forEach(model => {
            const option = document.createElement('div');
            option.className = 'px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer text-sm';
            option.textContent = model.name;
            option.dataset.modelId = model.id;
            option.dataset.modelName = model.name;
            
            option.addEventListener('click', () => {
                selectedModel = model.name;
                document.getElementById('model-display-text').textContent = model.name;
                dropdown.classList.add('hidden');
            });
            
            dropdown.appendChild(option);
        });
    }

    // 填充数据集下拉框
    function populateDatasetDropdown(datasets) {
        const dropdown = document.getElementById('dataset-dropdown');
        dropdown.innerHTML = '';
        
        datasets.forEach(dataset => {
            const option = document.createElement('div');
            option.className = 'px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer text-sm';
            option.textContent = dataset.名称;
            option.dataset.datasetId = dataset.id;
            option.dataset.datasetName = dataset.名称;
            
            option.addEventListener('click', () => {
                selectedDataset = dataset.id;
                document.getElementById('dataset-display-text').textContent = dataset.名称;
                dropdown.classList.add('hidden');
            });
            
            dropdown.appendChild(option);
        });
    }

    // 模型选择按钮点击事件
    document.getElementById('model-select-btn').addEventListener('click', function(e) {
        e.stopPropagation();
        const dropdown = document.getElementById('model-dropdown');
        dropdown.classList.toggle('hidden');
        
        // 隐藏数据集下拉框
        document.getElementById('dataset-dropdown').classList.add('hidden');
    });

    // 数据集选择按钮点击事件
    document.getElementById('dataset-select-btn').addEventListener('click', function(e) {
        e.stopPropagation();
        const dropdown = document.getElementById('dataset-dropdown');
        dropdown.classList.toggle('hidden');
        
        // 隐藏模型下拉框
        document.getElementById('model-dropdown').classList.add('hidden');
    });

    // 点击页面其他地方隐藏下拉框
    document.addEventListener('click', function(e) {
        if (!e.target.closest('#model-select-btn') && !e.target.closest('#model-dropdown')) {
            document.getElementById('model-dropdown').classList.add('hidden');
        }
        if (!e.target.closest('#dataset-select-btn') && !e.target.closest('#dataset-dropdown')) {
            document.getElementById('dataset-dropdown').classList.add('hidden');
        }
    });

    // 发送消息时使用当前选择的模型
    function getSelectedModel() {
        return selectedModel;
    }

    // 初始化加载数据
    loadModels();
    loadDatasets();

    // 创建消息容器块（区分thinking和result类型的样式）
    function createMessageBlock(type) {
        const block = document.createElement('div');
        block.className = `message-block ${type === 'thinking' ? 'thinking-block' : 'result-block'}`;
        return block;
    }

    function createMessageBlock(sender, isThinking = false, content = '') {
        if (!currentConversationContainer) {
            currentConversationContainer = document.createElement('div');
            currentConversationContainer.className = 'conversation-group border-b border-slate-200 dark:border-slate-800 pb-4 mb-4 last:border-0 last:mb-0';
            chatContainer.appendChild(currentConversationContainer);
        }
    
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const block = document.createElement('div');
        const userMessageClass = sender === 'user' ? 'user-message' : '';
        block.className = `flex flex-col py-2.5 message-block ${userMessageClass}`;
        block.dataset.messageId = messageId;
        block.dataset.sender = sender;
        block.dataset.isThinking = isThinking ? 'true' : 'false';
        
        let contentArea = '';
        if (isThinking) {
            contentArea = `
                <div class="thinking-content">
                    <div class="border-l-4 border-blue-300 bg-blue-50 dark:bg-blue-950/20 p-3 rounded-r-lg mb-2">
                        <div class="flex justify-between items-start">
                            <div class="thinking-text"></div>
                            <button class="collapse-thinking-btn text-blue-600 dark:text-blue-400 hover:text-blue-800 ml-2">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // 用户消息容器样式设置
            const contentClasses = sender === 'user' 
                ? 'user-content bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg text-right' 
                : 'result-content bg-slate-50 dark:bg-slate-800 p-3 rounded-lg';
            contentArea = `<div class="${contentClasses}">${content}</div>`;
        }
        
        // 调整内容块的整体对齐方式
        const flexDirection = sender === 'user' ? 'flex-row-reverse' : 'flex-row';
        const contentContainerClass = sender === 'user' 
            ? 'pe-10 flex justify-end'  
            : 'ps-10';                 
        const avatarMargin = sender === 'user' ? 'ml-2' : 'mr-2';
        const userNameAlign = sender === 'user' ? 'text-right' : '';
        
        block.innerHTML = `
            <div class="flex ${flexDirection} items-center gap-x-2">
                <div class="inline-flex flex-shrink-0 h-8 w-8 rounded-full overflow-hidden border-2 border-white dark:border-slate-700 ${avatarMargin}">
                    <img src="${sender === 'user' ? '/static/images/avatar/a.jpg' : '/static/images/avatar/bots/AAG.png'}" alt="" />
                </div>
                <h6 class="font-bold text-sm capitalize text-slate-600 dark:text-slate-100 ${userNameAlign}">
                    ${sender === 'user' ? 'you' : 'AAG'}${isThinking ? ' (thinking)' : ''}
                </h6>
            </div>
            <div class="${contentContainerClass} w-full">
                <div class="max-w-full text-slate-500 dark:text-slate-300 prose-strong:dark:text-white text-sm prose prose-code:![text-shadow:none] *:max-w-xl prose-pre:!max-w-full prose-pre:!w-full prose-pre:p-0 message-content">
                    ${contentArea}
                </div>
                ${sender === 'ai' && !isThinking ? `
                <div class="result-actions pt-4 flex gap-x-2 opacity-0 transition-opacity">
                    <button class="download-btn inline-flex justify-center items-center transition-all p-1 rounded-md bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:border-blue-500 hover:bg-blue-500 hover:dark:border-blue-500 hover:dark:bg-blue-500 hover:text-white hover:dark:text-white">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75v-2.25m-10.5-11.25h10.5a2.25 2.25 0 012.25 2.25v6.75a2.25 2.25 0 01-2.25 2.25H6.75a2.25 2.25 0 01-2.25-2.25v-6.75a2.25 2.25 0 012.25-2.25z" />
                        </svg>
                    </button>
                    <button class="regenerate-btn inline-flex justify-center items-center transition-all p-1 rounded-md bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:border-blue-500 hover:bg-blue-500 hover:dark:border-blue-500 hover:dark:bg-blue-500 hover:text-white hover:dark:text-white">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12c0-1.232-.046-2.453-.138-3.662a4.006 4.006 0 0 0-3.7-3.7 48.678 48.678 0 0 0-7.324 0 4.006 4.006 0 0 0-3.7 3.7c-.017.22-.032.441-.046.662M19.5 12l3-3m-3 3-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.006 4.006 0 0 0 3.7 3.7 48.656 48.656 0 0 0 7.324 0 4.006 4.006 0 0 0 3.7-3.7c.017-.22.032-.441.046-.662M4.5 12l3 3m-3-3-3 3" />
                        </svg>
                    </button>
                    <button class="feedback-btn inline-flex justify-center items-center transition-all p-1 rounded-md bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:border-blue-500 hover:bg-blue-500 hover:dark:border-blue-500 hover:dark:bg-blue-500 hover:text-white hover:dark:text-white">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715a12.137 12.137 0 0 1-.068-1.285c0-2.848.992-5.464 2.649-7.521C5.287 4.247 5.886 4 6.504 4h4.016a4.5 4.5 0 0 1 1.423.23l3.114 1.04a4.5 4.5 0 0 0 1.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 0 0 7.5 19.75 2.25 2.25 0 0 0 9.75 22a.75.75 0 0 0 .75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 0 0 2.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384m-10.253 1.5H9.7m8.075-9.75c.01.05.027.1.05.148.593 1.2.925 2.55.925 3.977 0 1.487-.36 2.89-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398-.306.774-1.086 1.227-1.918 1.227h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 0 0 .303-.54" />
                        </svg>
                    </button>
                    <button class="share-btn inline-flex justify-center items-center font-medium transition-all text-xs px-2 py-1 gap-2 rounded-md bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:border-green-500 hover:bg-green-500 hover:dark:border-green-500 hover:dark:bg-green-500 hover:text-white hover:dark:text-white">Share</button>
                </div>
                ` : ''}
            </div>
        `;
        
        currentConversationContainer.appendChild(block);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        if (sender === 'ai' && isThinking) {
            currentThinkingBlock = block;
            currentMessageId = messageId;

            const collapseBtn = block.querySelector('.collapse-thinking-btn');
            collapseBtn.addEventListener('click', toggleThinkingBlock);
        } else if (sender === 'ai' && !isThinking) {
            currentResultBlock = block;
        }
        
        return block;
    }

    // 处理Yes按钮点击事件
    function handleYesClick() {
        if (!isDagModeEnabled) return;
        if (isProcessing) return;
        
        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        currentConversationContainer = null;

        isProcessing = true;
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 animate-spin"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="30" stroke-dashoffset="0"/></svg>';
        
        // 创建AI思考过程消息
        createMessageBlock('ai', true);
        
        // 发送确认信息到后端（使用API获取数据）
        const selectedModel = getSelectedModel();
        const eventSource = new EventSource(`/api/chat?dag_confirm=yes&dag_id=${currentDagId}&model=${encodeURIComponent(selectedModel)}`);
        
        setupEventSourceHandlers(eventSource, false);
    }

    // 处理No按钮点击事件
    function handleNoClick() {
        if (!isDagModeEnabled) return;
        isDagReviewMode = true;
        
        chatInput.focus();
        // 显示提示文字
        chatInput.setAttribute('placeholder', 'Please enter your modifications for the DAG...');
    }

    function renderDag(dagData, parentContainer) {
        const dagWrapper = document.createElement('div');
        dagWrapper.className = 'my-4 p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 flex flex-col items-center';
        dagWrapper.style.textAlign = 'center'; 
        currentDagId = dagData.id || `dag-${Date.now()}`;
        currentDagContainer = dagWrapper;
        const mermaidId = `mermaid-dag-${Date.now()}`;
        const mermaidContainer = document.createElement('div');
        mermaidContainer.className = 'mermaid max-w-full overflow-x-auto';
        mermaidContainer.id = mermaidId;
        mermaidContainer.style.margin = '0 auto'; 
        mermaidContainer.style.display = 'inline-block';
        
        // 生成mermaid代码
        let mermaidCode = 'graph TD;\n';
        dagData.nodes.forEach(node => {
            const escapedLabel = node.label
                .replace(/"/g, '\\"')
                .replace(/;/g, ',')
                .replace(/-/g, '\\-');
            mermaidCode += `  ${node.id}["${escapedLabel}"];\n`;
        });
        dagData.edges.forEach(edge => {
            mermaidCode += `  ${edge.from} --> ${edge.to};\n`;
        });
        mermaidContainer.textContent = mermaidCode;
        dagWrapper.appendChild(mermaidContainer);
        
        // 只有在DAG模式开启时才显示按钮和示例
        if (isDagModeEnabled) {
            // 添加按钮
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'mt-4 flex gap-4';
            buttonContainer.innerHTML = `
                <button class="dag-yes-btn px-4 py-2 rounded-md transition-colors">生成答案</button>
                <button class="dag-no-btn px-4 py-2 rounded-md transition-colors">修改DAG</button>
            `;
            dagWrapper.appendChild(buttonContainer);
            buttonContainer.querySelector('.dag-yes-btn').addEventListener('click', handleYesClick);
            buttonContainer.querySelector('.dag-no-btn').addEventListener('click', handleNoClick);
            
            // 添加示例修改建议
            const exampleDiv = document.createElement('div');
            exampleDiv.className = 'text-sm text-slate-500 dark:text-slate-400 mb-3';
            exampleDiv.innerHTML =
                "<br>" +
                "<strong>示例修改建议：</strong><br><br>" +
                "节点修改方式：<br>" +  // 用 <br> 强制换行
                "调整现有节点标签：对原有节点的描述进行细化（如补充功能细节、限定范围、明确输出要求等），让节点作用更具体；<br>" +
                "新增节点：根据流程需求，在现有节点之间添加中间环节（如过滤、验证、转换等功能节点），完善逻辑链条；<br>" +
                "删除节点：移除流程中冗余、重复或不必要的节点，简化链路。<br><br>" +
                "连接关系修改方式：<br>" +
                "新增连接：在需要关联的节点间建立新的指向关系（如新增节点与前后节点的衔接、补充反馈链路等）；<br>" +
                "删除连接：移除无效的循环、冗余或逻辑矛盾的连接（如无意义的回环、重复指向等）；<br>" +
                "调整连接指向：改变原有连接的起点或终点，优化流程顺序（如将 \"检索→问题\" 的循环改为 \"检索→生成\" 的直接指向）。";
            dagWrapper.appendChild(exampleDiv);
        }
        
        parentContainer.appendChild(dagWrapper);
        
        // 创建Tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'dag-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 9999; /* 确保在最上层 */
            opacity: 0;
            transition: opacity 0.2s;
            white-space: nowrap; /* 防止ID换行 */
        `;
        document.body.appendChild(tooltip);
        
        // 渲染图表并绑定事件
        mermaid.init(undefined, `#${mermaidId}`).then(() => {
            const svgElement = document.querySelector(`#${mermaidId} svg`);
            if (!svgElement) {
                console.error('未找到SVG元素');
                return;
            }
            svgElement.style.margin = '0 auto';
            svgElement.style.display = 'block';
            
            // 存储节点ID与节点数据的映射（方便快速查找）
            const nodeMap = {};
            dagData.nodes.forEach(node => {
                nodeMap[node.id] = node;
            });
            
            // 调试：打印所有节点ID，确认数据正确
            console.log('DAG节点ID列表:', Object.keys(nodeMap));
            
            // 事件委托：监听整个SVG的鼠标事件（兼容性更好）
            svgElement.addEventListener('mouseover', (e) => {
                // 查找当前鼠标所在的节点元素（mermaid的节点通常是g元素，包含rect和text）
                const nodeG = e.target.closest('g[class*="node"]'); // 匹配包含"node"类的g元素
                if (!nodeG) return;
                // 从节点元素的ID中提取原始nodeId（兼容不同mermaid版本的ID格式）
                // mermaid可能生成的ID格式：node-xxx、flowchart-node-xxx等，需要提取核心ID
                const nodeGId = nodeG.id;
                let matchedNodeId = null;
                // 遍历节点ID，查找与当前元素ID部分匹配的（如nodeId在元素ID中出现）
                Object.keys(nodeMap).forEach(originalId => {
                    if (nodeGId.includes(originalId)) {
                        matchedNodeId = originalId;
                    }
                });
                if (matchedNodeId) {
                    // 显示Tooltip
                    tooltip.textContent = `节点ID: ${matchedNodeId}`;
                    // 定位在鼠标右下方
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY + 10}px`;
                    tooltip.style.opacity = '1';
                    console.log('悬停节点ID:', matchedNodeId); // 调试用
                }
            });
            
            // 鼠标移动时更新Tooltip位置
            svgElement.addEventListener('mousemove', (e) => {
                if (tooltip.style.opacity === '1') {
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY + 10}px`;
                }
            });
            
            // 鼠标离开SVG时隐藏Tooltip
            svgElement.addEventListener('mouseout', () => {
                tooltip.style.opacity = '0';
            });
        }).catch(err => {
            console.error('DAG渲染失败:', err);
            mermaidContainer.innerHTML = '<p class="text-red-500">图表渲染失败，请刷新页面重试</p>';
        });
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // 内容追加函数
    function appendContentToBlock(block, contentType, content) {
        // 根据消息类型确定内容应该添加到哪个容器
        let targetContainer;
        if (block.dataset.isThinking === 'true') {
            targetContainer = block.querySelector('.thinking-text');
        } else {
            targetContainer = block.querySelector('.result-content') || block.querySelector('.user-content');
        }

        if (!targetContainer) return;

        if (contentType === 'text') {
            // 处理文本内容
            const textElement = document.createElement('div');
            textElement.innerHTML = escapeHtml(content).replace(/\n/g, '<br>');
            targetContainer.appendChild(textElement);
        } else if (contentType === 'code') {
            // 处理代码内容（深色代码框）
            if (!content || !content.code) return;

            const codeBlock = document.createElement('div');
            codeBlock.className = 'code-container mt-2 mb-2';
            codeBlock.innerHTML = `
                <div class="code-header bg-slate-800 px-4 py-2 flex justify-between items-center rounded-t-lg">
                    <span class="text-xs text-slate-300 font-mono">${content.language || 'code'}</span>
                    <button class="copy-btn text-slate-300 hover:text-white text-xs">Copy</button>
                </div>
                <pre class="code-body bg-slate-900 p-4 rounded-b-lg overflow-x-auto"><code class="font-mono text-sm text-white">${escapeHtml(content.code)}</code></pre>
            `;
            targetContainer.appendChild(codeBlock);

            // 绑定复制按钮功能
            const copyBtn = codeBlock.querySelector('.copy-btn');
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(content.code);
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            });
        }
        else{
            // 如果是DAG类型，渲染DAG并显示按钮
            renderDag(content,targetContainer);

            if (!isDagModeEnabled) {
                console.log('Export Mode已关闭，仅显示图表');
            }
        }

        // 自动滚动到底部
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // HTML转义函数
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // 思考过程折叠函数
    function toggleThinkingBlock() {
        if (!currentThinkingBlock) return;
        
        const thinkingContent = currentThinkingBlock.querySelector('.thinking-content');
        const collapseBtn = currentThinkingBlock.querySelector('.collapse-thinking-btn');
        const thinkingText = currentThinkingBlock.querySelector('.thinking-text');
        
        if (thinkingContent && thinkingText) {
            // 检查当前是否处于折叠状态
            const isCollapsed = thinkingText.querySelector('.text-blue-600') !== null;
            
            if (isCollapsed) {
                // 展开思考过程
                const summary = thinkingText.querySelector('.text-blue-600');
                if (summary && summary.dataset.originalContent) {
                    thinkingText.innerHTML = summary.dataset.originalContent;
                    collapseBtn.innerHTML = `
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" />
                        </svg>
                    `;
                }
            } else {
                // 折叠思考过程
                const originalContent = thinkingText.innerHTML;
                
                // 创建摘要
                const summary = document.createElement('div');
                summary.className = 'text-blue-600 dark:text-blue-400 cursor-pointer hover:underline';
                summary.textContent = 'Show thinking process...';
                summary.dataset.originalContent = originalContent;
                
                // 替换内容
                thinkingText.innerHTML = '';
                thinkingText.appendChild(summary);
                
                // 更新折叠按钮
                collapseBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                    </svg>
                `;
                
                // 添加展开事件
                summary.addEventListener('click', toggleThinkingBlock);
            }
        }
    }
    
    function downloadTextAsFile(text, filename) {
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // 清理临时资源
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
    }

    // 收集当前回答的内容（不包括思考过程）
    function collectAiResponseContent(resultBlock) {
        const contentContainer = resultBlock.querySelector('.result-content');
        if (!contentContainer) return '';
        
        let textContent = '';
        
        // 收集文本内容
        const textElements = contentContainer.querySelectorAll('div:not(.code-container)');
        textElements.forEach(el => {
            textContent += el.textContent + '\n\n';
        });
        
        // 收集代码内容
        const codeBlocks = contentContainer.querySelectorAll('.code-container');
        codeBlocks.forEach(block => {
            const language = block.querySelector('.code-header span').textContent;
            const code = block.querySelector('.code-body code').textContent;
            
            textContent += `[${language}代码]\n`;
            textContent += code + '\n\n';
        });
        
        return textContent.trim();
    }

    // 获取选中模型的函数
    function getSelectedModel() {
        for (const radio of modelRadios) {
            if (radio.checked) {
                return radio.value;
            }
        }
        return 'GPT 4'; // 默认模型
    }

    // 设置EventSource的处理函数
    function setupEventSourceHandlers(eventSource, isDagModification = false) {
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                console.log('Received data:', data);
                
                if (data.error) {
                    throw new Error(data.error);
                }
            
                if (data.type === 'thinking') {
                    appendContentToBlock(currentThinkingBlock, data.contentType, data.content);
                } else if (data.type === 'result') {
                    // 如果是第一条最终结果，先折叠思考过程
                    if (!currentResultBlock) {
                        toggleThinkingBlock();
                        // 创建最终结果消息
                        currentResultBlock = createMessageBlock('ai');
                    }
                    
                    appendContentToBlock(currentResultBlock, data.contentType, data.content);
                    
                    // 如果收到的是新的DAG，保持审核模式
                    if (data.contentType !== 'dag') {
                        isDagReviewMode = false;
                        chatInput.setAttribute('placeholder', 'Type your message...');
                    }
                }
            } catch (error) {
                console.error('Error processing message:', error);
                alert('Error processing response: ' + error.message);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('EventSource error:', error);
            eventSource.close();
            resetSendButton();
        };
        
        eventSource.onopen = function() {
            console.log('Connection opened');
        };
        
        // 当流结束时
        eventSource.addEventListener('end', function() {
            console.log('Stream ended');
            eventSource.close();
            resetSendButton();
            
            // 显示操作按钮
            if (currentResultBlock) {
                const actions = currentResultBlock.querySelector('.result-actions');
                if (actions) {
                    setTimeout(() => {
                        actions.classList.remove('opacity-0');
                    }, 300);
                }
            }
        });
        
        // 当发生错误时
        eventSource.addEventListener('error', function(event) {
            console.error('Stream error:', event);
            eventSource.close();
            resetSendButton();
        });
    }

    // 发送消息函数（使用EventSource）
    function sendMessageWithEventSource() {
        const message = chatInput.textContent.trim();
        if (!message && !isDagReviewMode || isProcessing) return;
        
        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        currentConversationContainer = null; 
        isProcessing = true;
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 animate-spin"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="30" stroke-dashoffset="0"/></svg>';
        
        // 清空输入框
        chatInput.textContent = '';
        adjustInputHeight();
        
        // 如果是DAG修改模式，但DAG模式已关闭，直接发送消息而不进入修改流程
        if (isDagReviewMode && !isDagModeEnabled) {
            // 重置DAG相关状态
            isDagReviewMode = false;
            currentDagId = null;
        }
        
        // 创建用户消息并填充内容
        const userMessage = isDagReviewMode ? 
            `I don't confirm this DAG (No). Modifications: ${message}` : 
            message;
        createMessageBlock('user', false, userMessage);
        
        // 创建思考过程消息
        createMessageBlock('ai', true);
        
        // 发送请求到后端
        const selectedModel = getSelectedModel();
        let url = `/api/chat?model=${encodeURIComponent(selectedModel)}`;
        if (selectedDataset) {
            url += `&dataset=${encodeURIComponent(selectedDataset)}`;
        }
        
        if (isDagReviewMode) {
            // 如果是DAG修改模式，发送no确认和修改意见
            url += `&dag_confirm=no&dag_id=${currentDagId}&modifications=${encodeURIComponent(message)}`;
        } else {
            // 普通消息
            url += `&message=${encodeURIComponent(message)}`;
        }
        
        const eventSource = new EventSource(url);
        setupEventSourceHandlers(eventSource, isDagReviewMode);
    }

    // 辅助函数：重置发送按钮状态
    function resetSendButton() {
        isProcessing = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" /></svg>';
    }

    // 辅助函数：调整输入框高度
    function adjustInputHeight() {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    }

    // 统一的发送消息入口
    function sendMessage() {
        sendMessageWithEventSource();
    }

    // 绑定事件
    // 发送按钮点击
    sendBtn.addEventListener('click', sendMessage);
    
    // 输入框按Enter发送（Shift+Enter换行）
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 输入框高度自适应
    chatInput.addEventListener('input', adjustInputHeight);
    
    // 初始化输入框高度
    adjustInputHeight();

    // 为代码块添加复制功能
    document.addEventListener('click', function(e) {

        if (e.target.closest('.collapse-thinking-btn')) {
            e.preventDefault();
            toggleThinkingBlock();
            return;
        }

        if (e.target.classList.contains('js-copy')) {
            const codeBlock = e.target.closest('.js-code-block');
            const codeElement = codeBlock.querySelector('code');
            const textToCopy = codeElement.textContent;
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = e.target.textContent;
                e.target.textContent = 'Copied!';
                e.target.classList.add('success');
                
                setTimeout(() => {
                    e.target.textContent = originalText;
                    e.target.classList.remove('success');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        }

        if (e.target.closest('.download-btn')) {
            e.preventDefault();
            const resultBlock = e.target.closest('.message-block');
            const content = collectAiResponseContent(resultBlock);
        
            if (content) {
                // 生成包含时间戳的文件名
                const now = new Date();
                const timestamp = now.toISOString().replace(/[:.]/g, '-');
                const filename = `ai-response-${timestamp}.txt`;
                
                downloadTextAsFile(content, filename);
            } else {
                alert('没有可下载的内容');
            }
        }
    });

    console.log('聊天界面初始化完成');
});
