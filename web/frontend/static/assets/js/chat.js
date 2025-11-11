document.addEventListener('DOMContentLoaded', function() {
    // 通用DOM元素
    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const modelRadios = document.querySelectorAll('input[name="language-model"]');
    
    let currentThinkingBlock = null;
    let currentResultBlock = null;
    let currentMessageId = null;
    let isProcessing = false;
    let currentConversationContainer = null;

    // 处理消息的核心函数（接收消息数组并渲染）
    function renderMessages(messages) {
        messages.forEach(message => {
            // 根据消息类型（thinking/result）创建对应的容器块
            const block = createMessageBlock(message.type);
            chatContainer.appendChild(block);

            // 渲染当前消息项的内容（文本/代码）
            appendContentToBlock(block, message.contentType, message.content);
        });
    }

    // 创建消息容器块（区分thinking和result类型的样式）
    function createMessageBlock(type) {
        const block = document.createElement('div');
        block.className = `message-block ${type === 'thinking' ? 'thinking-block' : 'result-block'}`;
        // 可根据类型添加不同样式（如thinking块用浅蓝色背景）
        return block;
    }

    function createMessageBlock(sender, isThinking = false, content = '') {

        // 若没有当前对话容器，创建一个新的（用于分组单轮对话）
        if (!currentConversationContainer) {
            currentConversationContainer = document.createElement('div');
            currentConversationContainer.className = 'conversation-group border-b border-slate-200 dark:border-slate-800 pb-4 mb-4 last:border-0 last:mb-0'; // 添加分隔线和间距
            chatContainer.appendChild(currentConversationContainer);
        }
        
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const block = document.createElement('div');
        block.className = 'flex flex-col py-2.5 message-block';
        block.dataset.messageId = messageId;
        block.dataset.sender = sender;
        block.dataset.isThinking = isThinking ? 'true' : 'false';
        
        // 消息内容区域结构（根据发送者和类型调整）
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
            // 用户消息直接显示内容，AI消息留空后续填充
            contentArea = `<div class="${sender === 'user' ? 'user-content' : 'result-content'}">${content}</div>`;
        }
        
        block.innerHTML = `
            <div class="flex items-center gap-x-2">
                <div class="inline-flex flex-shrink-0 h-8 w-8 rounded-full overflow-hidden border-2 border-white dark:border-slate-700">
                    <img src="${sender === 'user' ? '/static/images/avatar/a.jpg' : '/static/images/avatar/bots/AAG.png'}" alt="" />
                </div>
                <h6 class="font-bold text-sm capitalize text-slate-600 dark:text-slate-100">
                    ${sender === 'user' ? 'you' : 'AAG'}${isThinking ? ' (thinking)' : ''}
                </h6>
            </div>
            <div class="ps-10 w-full">
                <div class="max-w-full text-slate-500 dark:text-slate-300 prose-strong:dark:text-white text-sm prose prose-code:![text-shadow:none] *:max-w-xl prose-pre:!max-w-full prose-pre:!w-full prose-pre:p-0 message-content">
                    ${contentArea}
                </div>
                ${sender === 'ai' && !isThinking ? `
                <div class="result-actions pt-4 flex gap-x-2 opacity-0 transition-opacity">
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

    // 获取选中模型的函数
    function getSelectedModel() {
        for (const radio of modelRadios) {
            if (radio.checked) {
                return radio.value;
            }
        }
        return 'GPT 4'; // 默认模型
    }

    // 发送消息函数（使用EventSource）
    function sendMessageWithEventSource() {
        const message = chatInput.textContent.trim();
        if (!message || isProcessing) return;
        
        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        currentConversationContainer = null; // 重置当前对话容器，开始新的对话组

        isProcessing = true;
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 animate-spin"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="30" stroke-dashoffset="0"/></svg>';
        
        // 清空输入框
        chatInput.textContent = '';
        adjustInputHeight();
        
        // 创建用户消息并填充内容
        createMessageBlock('user', false, message);
        
        // 创建思考过程消息
        createMessageBlock('ai', true);
        
        // 发送请求到后端
        const selectedModel = getSelectedModel();
        
        const eventSource = new EventSource(`/api/chat?message=${encodeURIComponent(message)}&model=${encodeURIComponent(selectedModel)}`);
        
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
    });

    console.log('聊天界面初始化完成');
});
