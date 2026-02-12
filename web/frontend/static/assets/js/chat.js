document.addEventListener('DOMContentLoaded', function() {
    const socket = io(); 

    // --- 国际化辅助函数 ---
    // 获取翻译文本，如果未加载则返回默认值
    function t(key, defaultVal) {
        if (window.LANG && window.LANG[key]) {
            return window.LANG[key];
        }
        return defaultVal || key;
    }

    // 更新当前页面上已存在的 JS 生成元素的文本
    function updateJSTranslations() {
        // 1. 更新带有 data-i18n 属性的常规元素 (如按钮、提示文本)
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (key) {
                // 如果是 INPUT，更新 placeholder
                if (el.tagName === 'INPUT') {
                    el.placeholder = t(key, el.placeholder);
                } 
                // 如果是 ContentEditable DIV (聊天输入框)
                else if (el.id === 'chat-input') {
                    el.setAttribute('placeholder', t(key, el.getAttribute('placeholder')));
                }
                // 普通文本
                else {
                    el.innerText = t(key, el.innerText);
                }
            }
        });

        // 2. 更新聊天记录中的名字和状态
        // 这里的元素没有 data-i18n 属性，所以需要通过类名 .sender-name 单独选取
        document.querySelectorAll('.sender-name').forEach(el => {
            const senderKey = el.dataset.senderKey; // 'sender_you' 或 'sender_ai'
            const isThinking = el.dataset.thinking === 'true';
            
            // 获取名字翻译
            // 如果 t() 返回 key 本身 (说明没找到翻译)，且 key 是 sender_ai，强制显示 'AAG'
            let displayName = t(senderKey);
            if (displayName === senderKey && senderKey === 'sender_ai') {
                displayName = 'AAG';
            }

            // 获取状态后缀翻译
            const thinkingSuffix = isThinking ? t('status_thinking', ' (Thinking...)') : '';
            
            // 更新 HTML
            el.innerHTML = displayName + thinkingSuffix;
        });

        // 3. 更新输入框 Placeholder (兜底)
        const chatInput = document.getElementById('chat-input');
        if(chatInput) {
            chatInput.setAttribute('placeholder', t('input_placeholder', 'Type your message...'));
        }
    }

    // 监听 HTML 页面发出的语言加载完成事件
    document.addEventListener('languageLoaded', () => {
        updateJSTranslations();
        // 重新渲染侧边栏标题（如果是默认标题）
        // 这里略过，保持用户输入的一致性
    });
    // -----------------------

    mermaid.initialize({
        startOnLoad: false,
        theme: 'neutral',
        flowchart: {
            curve: 'basis',         
            nodeSpacing: 50,
            rankSpacing: 80
        },
        securityLevel: 'loose'
    });

    let chatSessions = {}; 
    let activeSessionId = null; 

    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const conversationList = document.querySelector('#chatAside ul');

    // Storage Keys
    const STORAGE_KEY = 'chat_sessions_data_v1';
    const ACTIVE_SESSION_KEY = 'chat_active_session_id_v1';

    const SEND_BTN_SPINNING = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4 animate-spin">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-opacity="0.3" stroke-width="2" fill="none"/>
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" stroke-width="2"/>
        </svg>
    `;

    const SEND_BTN_NORMAL = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
        </svg>
    `;
    
    let currentThinkingBlock = null;
    let currentResultBlock = null;
    let currentMessageId = null;
    let isProcessing = false;
    let currentConversationContainer = null;
    let currentDagContainer = null; 
    let isDagReviewMode = false; 
    let currentDagId = null; 
    let isDagModeEnabled = false;
    let isInteractiveModeEnabled = false;
    let selectedModel = 'GPT 4'; 
    let selectedDataset = null; 

    // Session Management Functions
    function saveState() {
        if (isProcessing) return; 
        
        const sessionData = {};
        for (const [id, session] of Object.entries(chatSessions)) {
            sessionData[id] = {
                id: session.id,
                title: session.title,
                timestamp: session.timestamp,
                messages: session.messages || [] 
            };
        }
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(sessionData));
        if (activeSessionId) {
            sessionStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
        }
    }

    function forceSaveState() {
        const sessionData = {};
        for (const [id, session] of Object.entries(chatSessions)) {
            sessionData[id] = {
                id: session.id,
                title: session.title,
                timestamp: session.timestamp,
                messages: session.messages || []
            };
        }
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(sessionData));
        if (activeSessionId) {
            sessionStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
        }
    }

    // 初始化交互式模式状态
    if (localStorage.getItem('interactiveModeEnabled') !== null) {
        isInteractiveModeEnabled = localStorage.getItem('interactiveModeEnabled') === 'true';
        const interactiveToggle = document.getElementById('interactive-mode-toggle');
        if (interactiveToggle) {
            interactiveToggle.checked = isInteractiveModeEnabled;
        }
    }

    // 监听交互式模式切换
    const expertToggle = document.getElementById('dag-mode-toggle');
    const interactiveToggle = document.getElementById('interactive-mode-toggle');

    // 1. 专家模式 (Expert Mode) 监听器
    if (expertToggle) {
        expertToggle.addEventListener('change', function(e) {
            isDagModeEnabled = e.target.checked;
            
            // 互斥逻辑：如果开启了专家模式，强制关闭交互式模式
            if (isDagModeEnabled) {
                isInteractiveModeEnabled = false;
                if (interactiveToggle) interactiveToggle.checked = false;
                localStorage.setItem('interactiveModeEnabled', 'false');
            }
            
            localStorage.setItem('dagModeEnabled', isDagModeEnabled);
        });
    }

    // 2. 交互式模式 (Interactive Mode) 监听器
    if (interactiveToggle) {
        interactiveToggle.addEventListener('change', function(e) {
            isInteractiveModeEnabled = e.target.checked;
            
            // 互斥逻辑：如果开启了交互式模式，强制关闭专家模式
            if (isInteractiveModeEnabled) {
                isDagModeEnabled = false;
                if (expertToggle) expertToggle.checked = false;
                localStorage.setItem('dagModeEnabled', 'false');
            }
            
            localStorage.setItem('interactiveModeEnabled', isInteractiveModeEnabled);
        });
    }
    
    // 3. 初始化状态加载 
    if (localStorage.getItem('dagModeEnabled') === 'true') {
        isDagModeEnabled = true;
        isInteractiveModeEnabled = false; // 确保互斥
        if (expertToggle) expertToggle.checked = true;
        if (interactiveToggle) interactiveToggle.checked = false;
    } else if (localStorage.getItem('interactiveModeEnabled') === 'true') {
        isInteractiveModeEnabled = true;
        isDagModeEnabled = false; 
        if (interactiveToggle) interactiveToggle.checked = true;
        if (expertToggle) expertToggle.checked = false;
    }

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

    function scrollToBottom(smooth = false) {
        requestAnimationFrame(() => {
            setTimeout(() => {
                if (chatContainer) {
                    const targetScroll = chatContainer.scrollHeight - chatContainer.clientHeight;
                    if (Math.abs(chatContainer.scrollTop - targetScroll) > 10) { 
                        if (smooth) {
                            chatContainer.scrollTo({
                                top: chatContainer.scrollHeight,
                                behavior: 'smooth'
                            });
                        } else {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    }
                }
            }, 60);
        });
    }

    function populateModelDropdown(models) {
        const dropdown = document.getElementById('model-dropdown');
        dropdown.innerHTML = '';
        
        // 1. 渲染现有的模型列表
        models.forEach(model => {
            const option = document.createElement('div');
            option.className = 'px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer text-sm';
            option.textContent = model.name;
            option.addEventListener('click', () => {
                selectedModel = model.name;
                document.getElementById('model-display-text').textContent = model.name;
                dropdown.classList.add('hidden');
            });
            dropdown.appendChild(option);
        });

        // 2. 添加分隔线 (美观)
        const separator = document.createElement('div');
        separator.className = 'border-t border-slate-200 dark:border-slate-800 my-1 mx-2';
        dropdown.appendChild(separator);

        // 3. 添加配置跳转按钮 (+)
        const configBtn = document.createElement('div');
        // 使用 text-blue-600 让它看起来像是一个操作按钮，而不是普通选项
        configBtn.className = 'px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer text-sm text-blue-600 dark:text-blue-400 flex items-center gap-2 font-medium';
        
        // 使用 t() 函数支持国际化，如果没找到翻译则显示 'Configure Models'
        const btnText = typeof t === 'function' ? t('btn_config_models', 'Configure Models') : 'Configure Models';
        
        configBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            <span>${btnText}</span>
        `;
        
        configBtn.addEventListener('click', (e) => {
            e.stopPropagation(); 
            window.location.href = "/model_manager";
        });

        dropdown.appendChild(configBtn);
    }

    
    function populateDatasetDropdown(datasets) {
        const dropdown = document.getElementById('dataset-dropdown');
        dropdown.innerHTML = '';
        datasets.forEach(dataset => {
            const option = document.createElement('div');
            option.className = 'px-4 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer text-sm';
            option.textContent = dataset.名称;
            option.addEventListener('click', () => {
                selectedDataset = dataset.id;
                document.getElementById('dataset-display-text').textContent = dataset.名称;
                dropdown.classList.add('hidden');
            });
            dropdown.appendChild(option);
        });
    }

     document.getElementById('model-select-btn').addEventListener('click', function(e) {
        e.stopPropagation();
        document.getElementById('model-dropdown').classList.toggle('hidden');
        document.getElementById('dataset-dropdown').classList.add('hidden');
    });

    document.getElementById('dataset-select-btn').addEventListener('click', function(e) {
        e.stopPropagation();
        document.getElementById('dataset-dropdown').classList.toggle('hidden');
        document.getElementById('model-dropdown').classList.add('hidden');
    });

    document.addEventListener('click', function(e) {
        if (!e.target.closest('#model-select-btn') && !e.target.closest('#model-dropdown')) {
            document.getElementById('model-dropdown').classList.add('hidden');
        }
        if (!e.target.closest('#dataset-select-btn') && !e.target.closest('#dataset-dropdown')) {
            document.getElementById('dataset-dropdown').classList.add('hidden');
        }
    });

    function getSelectedModel() {
        return selectedModel;
    }

    loadModels();
    loadDatasets();

    socket.on('chat_response', function(data) {
        if (!isProcessing && data.type !== 'stream_end' && !data.error) return;

        try {
            if (data.error) {
                console.error('WebSocket Error:', data.error);
                alert(t('server_error', 'Server Error: ') + data.error);
                resetSendButton();
                return;
            }

            if (data.type === 'stream_end') {
                console.log('Stream ended via WebSocket');
                resetSendButton();
                
                if (currentResultBlock) {
                    const actions = currentResultBlock.querySelector('.result-actions');
                    if (actions) {
                        setTimeout(() => actions.classList.remove('opacity-0'), 300);
                    }
                }

                if (isDagReviewMode) {
                    exitDagModificationMode();
                }
                isDagReviewMode = false;
                chatInput.setAttribute('placeholder', t('input_placeholder', 'Type your message...'));
                
                forceSaveState();
                return;
            }

            if (data.type === 'thinking') {
                if (!currentThinkingBlock) {
                    createMessageBlock('ai', true); 
                }
                appendContentToBlock(currentThinkingBlock, data.contentType, data.content);
            } 
            else if (data.type === 'result') {
                if (!currentResultBlock) {
                    if (currentThinkingBlock) toggleThinkingBlock(); 
                    currentResultBlock = createMessageBlock('ai');
                    
                    if (data.contentType === 'dag') {
                        currentResultBlock.dataset.dagId = currentDagId || `dag-${Date.now()}`;
                    }
                }
                appendContentToBlock(currentResultBlock, data.contentType, data.content);
            }
        } catch (error) {
            console.error('解析 WebSocket 消息失败:', error);
            resetSendButton();
        }
    });

    // Modified createMessageBlock to support State Restoration and i18n
    function createMessageBlock(sender, isThinking = false, content = '', saveToHistory = true) {
        if (!activeSessionId) {
             createNewConversation();
        }
        
        const currentSession = chatSessions[activeSessionId];
        if (!currentConversationContainer || !currentSession.container.contains(currentConversationContainer)) {
            currentConversationContainer = document.createElement('div');
            currentConversationContainer.className = 'conversation-group border-b border-slate-200 dark:border-slate-800 pb-4 mb-4 last:border-0 last:mb-0';
            currentSession.container.appendChild(currentConversationContainer);
        }
    
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const block = document.createElement('div');
        const userMessageClass = sender === 'user' ? 'user-message' : '';
        block.className = `flex flex-col py-2.5 message-block ${userMessageClass}`;
        block.dataset.messageId = messageId;
        block.dataset.sender = sender;  
        block.dataset.isThinking = isThinking ? 'true' : 'false';
        
        if (saveToHistory && activeSessionId) {
            const msgData = {
                id: messageId,
                sender: sender,
                isThinking: isThinking,
                content: content,
                contentType: 'text', 
                timestamp: new Date().toISOString()
            };
            if (!currentSession.messages) currentSession.messages = [];
            currentSession.messages.push(msgData);
            block.dataset.msgIndex = currentSession.messages.length - 1;
        }

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
            const contentClasses = sender === 'user' 
                ? 'user-content bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg text-right' 
                : 'result-content bg-slate-50 dark:bg-slate-800 p-3 rounded-lg';
            
            let displayContent = content;
            if (sender === 'user') {
                displayContent = escapeHtml(content);
            } else if (sender === 'ai' && content && typeof content === 'string') {
                 displayContent = marked.parse(content);
            } else if (typeof content === 'object') {
                displayContent = ''; // DAGs handled via renderDag call later
            }

            contentArea = `<div class="${contentClasses}">${displayContent}</div>`;
        }
        
        const flexDirection = sender === 'user' ? 'flex-row-reverse' : 'flex-row';
        const contentContainerClass = sender === 'user' ? 'pe-10 flex justify-end' : 'ps-10';                 
        const avatarMargin = sender === 'user' ? 'ml-2' : 'mr-2';
        const userNameAlign = sender === 'user' ? 'text-right' : '';
        
        // i18n logic for Name
        const senderKey = sender === 'user' ? 'sender_you' : 'sender_ai';
        const senderName = t(senderKey);
        const thinkingSuffix = isThinking ? t('status_thinking', ' (thinking...)') : '';
        
        block.innerHTML = `
            <div class="flex ${flexDirection} items-center gap-x-2">
                <div class="inline-flex flex-shrink-0 h-8 w-8 rounded-full overflow-hidden border-2 border-white dark:border-slate-700 ${avatarMargin}">
                    <img src="${sender === 'user' ? '/static/images/avatar/human.jpeg' : '/static/images/avatar/bots/AAG.png'}" alt="" />
                </div>
                <h6 class="sender-name font-bold text-sm capitalize text-slate-600 dark:text-slate-100 ${userNameAlign}" 
                    data-i18n-dynamic="true" 
                    data-sender-key="${senderKey}" 
                    data-thinking="${isThinking}">
                    ${senderName}${thinkingSuffix}
                </h6>
            </div>
            <div class="${contentContainerClass} w-full">
                <div class="max-w-full text-slate-500 dark:text-slate-300 prose-strong:dark:text-white text-sm prose prose-code:![text-shadow:none] *:max-w-xl prose-pre:!max-w-full prose-pre:!w-full prose-pre:p-0 message-content">
                    ${contentArea}
                </div>
                ${sender === 'ai' && !isThinking ? `
                <div class="result-actions pt-4 flex gap-x-2 opacity-0 transition-opacity">
                    <button class="download-btn inline-flex justify-center items-center transition-all p-1 rounded-md bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:border-blue-500 hover:bg-blue-500 hover:dark:border-blue-500 hover:dark:bg-blue-500 hover:text-white hover:dark:text-white">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m0 0l6.75-6.75M12 19.5l-6.75-6.75" />
                            <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 21h15" />
                        </svg>
                    </button>
                </div>
                ` : ''}
            </div>
        `;
        
        currentConversationContainer.appendChild(block);
        
        if (sender === 'ai' && isThinking) {
            currentThinkingBlock = block;
            currentMessageId = messageId;
            const collapseBtn = block.querySelector('.collapse-thinking-btn');
            collapseBtn.addEventListener('click', toggleThinkingBlock);
            
            if (!saveToHistory && content) {
                 const thinkingText = block.querySelector('.thinking-text');
                 if(thinkingText) {
                     thinkingText.innerHTML = content;
                     toggleThinkingBlockState(block, true);
                 }
            }
        } else if (sender === 'ai' && !isThinking) {
            currentResultBlock = block;
            if (!saveToHistory) {
                 const actions = block.querySelector('.result-actions');
                 if (actions) actions.classList.remove('opacity-0');
            }
        }
        
        scrollToBottom();
        return block;
    }

    // Helper to toggle thinking state specifically
    function toggleThinkingBlockState(block, forceCollapse = false) {
        const thinkingText = block.querySelector('.thinking-text');
        const collapseBtn = block.querySelector('.collapse-thinking-btn');
        if (!thinkingText || !collapseBtn) return;

        const originalContent = thinkingText.innerHTML;
        if (!originalContent) return; 

        if (forceCollapse) {
             const summary = document.createElement('div');
             summary.className = 'text-blue-600 dark:text-blue-400 cursor-pointer hover:underline';
             summary.textContent = t('show_thinking', 'Show thinking process...'); // i18n
             summary.setAttribute('data-i18n', 'show_thinking'); // Add marker
             summary.dataset.originalContent = originalContent;
             
             thinkingText.innerHTML = '';
             thinkingText.appendChild(summary);
             
             collapseBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
            `;
             summary.addEventListener('click', toggleThinkingBlock);
        }
    }

    function handleYesClick() {
        if (!isDagModeEnabled) return;
        if (isProcessing) return;
        
        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        currentConversationContainer = null;

        isProcessing = true;
        sendBtn.disabled = true;
        sendBtn.innerHTML = SEND_BTN_SPINNING;
        
        createMessageBlock('ai', true);
        
        const selectedModel = getSelectedModel();
        socket.emit('chat_request', {
            dag_confirm: 'yes',
            dag_id: currentDagId,
            model: selectedModel
        });
    }

    function handleNoClick() {
        if (!isDagModeEnabled) return;
        isDagReviewMode = true;
        
        const existingCancelButton = document.querySelector('.cancel-modification-btn');
        if (existingCancelButton && existingCancelButton.parentNode) {
            existingCancelButton.parentNode.removeChild(existingCancelButton);
        }
        
        const existingTooltip = document.querySelector('.dag-modification-tooltip');
        if (existingTooltip && existingTooltip.parentNode) {
            existingTooltip.parentNode.removeChild(existingTooltip);
        }
        
        const tooltipElement = document.createElement('div');
        tooltipElement.className = 'dag-modification-tooltip';
        // i18n applied here
        tooltipElement.innerHTML = `
            <div class="tooltip-content">
                <div class="tooltip-arrow"></div>
                <p class="tooltip-text" data-i18n="tooltip_dag_mod_input">${t('tooltip_dag_mod_input', 'Please enter your dag modification suggestions here (You can press ESC to cancel)')}</p>
                <button class="tooltip-close">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
        `;
        
        if (!document.getElementById('dag-tooltip-style')) {
            const style = document.createElement('style');
            style.id = 'dag-tooltip-style';
            style.textContent = `
                .dag-modification-tooltip { position: absolute; z-index: 10000; animation: fadeInUp 0.3s ease-out; }
                .tooltip-content { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 12px; position: relative; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); max-width: 400px; text-align: center; }
                .tooltip-arrow { position: absolute; bottom: -6px; left: 50%; transform: translateX(-50%); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 8px solid #667eea; }
                .tooltip-text { margin: 0 24px 0 0; font-size: 14px; line-height: 1.4; font-weight: 500; }
                .tooltip-close { position: absolute; right: 8px; top: 8px; background: rgba(255, 255, 255, 0.2); border: none; width: 24px; height: 24px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; }
                .tooltip-close:hover { background: rgba(255, 255, 255, 0.3); transform: rotate(90deg); }
                .tooltip-close svg { stroke: white; width: 16px; height: 16px; }
                @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
                @keyframes fadeOutDown { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(10px); } }
                .tooltip-fade-out { animation: fadeOutDown 0.2s ease-out forwards; }
            `;
            document.head.appendChild(style);
        }
        
        if (!chatInput) return;
        
        const inputRect = chatInput.getBoundingClientRect();
        const bodyRect = document.body.getBoundingClientRect();
        document.body.appendChild(tooltipElement);
        
        const tooltipRect = tooltipElement.getBoundingClientRect();
        tooltipElement.style.left = `${inputRect.left - bodyRect.left + (inputRect.width / 2) - (tooltipRect.width / 2)}px`;
        tooltipElement.style.top = `${inputRect.top - bodyRect.top - tooltipRect.height - 8}px`;
        
        const currentDagBlock = currentResultBlock || currentThinkingBlock;
        if (currentDagBlock) {
            currentDagId = currentDagBlock.dataset.dagId || `dag-${Date.now()}`;
        }
        
        document.addEventListener('keydown', handleEscKey);
        tooltipElement.querySelector('.tooltip-close').addEventListener('click', exitDagModificationMode);
        chatInput.focus();
    }

    function exitDagModificationMode() {
        isDagReviewMode = false;
        chatInput.setAttribute('placeholder', t('input_placeholder', 'Type your message...'));
        
        const tooltip = document.querySelector('.dag-modification-tooltip');
        if (tooltip) {
            tooltip.classList.add('tooltip-fade-out');
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.parentNode.removeChild(tooltip);
                }
            }, 200);
        }
        
        const cancelButton = document.querySelector('.cancel-modification-btn');
        if (cancelButton && cancelButton.parentNode) {
            cancelButton.parentNode.removeChild(cancelButton);
        }
        
        document.removeEventListener('keydown', handleEscKey);
    }

    function handleEscKey(e) {
        if (e.key === 'Escape' && isDagReviewMode) {
            exitDagModificationMode();
        }
    }

    function renderDag(dagData, parentContainer) {
        const dagWrapper = document.createElement('div');
        dagWrapper.className = 'dag-wrapper my-4';
        dagWrapper.style.textAlign = 'center'; 

        currentDagId = dagData.id || `dag-${Date.now()}`;
        currentDagContainer = dagWrapper;
        
        const mermaidId = `mermaid-dag-${Date.now()}-${Math.floor(Math.random()*1000)}`;
        const mermaidContainer = document.createElement('div');
        mermaidContainer.className = 'mermaid max-w-full overflow-x-auto';
        mermaidContainer.id = mermaidId;
        mermaidContainer.style.margin = '0 auto'; 
        mermaidContainer.style.display = 'inline-block';
        mermaidContainer.style.opacity = '0'; 
        mermaidContainer.style.transition = 'opacity 0.3s ease';
        
        let mermaidCode = 'graph TD;\n';
        mermaidCode += '    classDef default fill:#bfdbfe,stroke:#4f46e5,stroke-width:1.5px,rx:8px,ry:8px,text-align:left,font-size:13px,font-weight:500,color:#1e293b;\n';

        dagData.nodes.forEach(node => {
            const nodeId = node.id || 'unknown';
            let fullLabel = (node.label || 'Untitled Task').trim();
            if (!fullLabel) fullLabel = 'Untitled Task';

            let shortContent = fullLabel;
            
            const sentenceEnd = fullLabel.search(/[。？！?.！;；]/);
            if (sentenceEnd > 10 && sentenceEnd < 60) {
                shortContent = fullLabel.substring(0, sentenceEnd + 1);
            } else {
                const words = fullLabel.split(/\s+/);
                if (words.length > 4) {
                    shortContent = words.slice(0, 4).join(' ') + '...';
                } else if (fullLabel.length > 15) {
                    shortContent = fullLabel.substring(0, 15) + '...';
                }
            }

            const MAX_LABEL_LENGTH = 40; 
            let finalLabel = `${nodeId}: ${shortContent}`;
            
            if (finalLabel.length > MAX_LABEL_LENGTH) {
                const available = Math.max(0, MAX_LABEL_LENGTH - nodeId.length - 2); 
                if (shortContent.length > available) {
                    const cutLength = Math.max(0, available - 3);
                    shortContent = shortContent.substring(0, cutLength) + '...';
                }
                finalLabel = `${nodeId}: ${shortContent}`;
            }

            const escapedLabel = finalLabel.replace(/"/g, '\\"').replace(/]/g, '\\]').replace(/\n/g, ' ');
            mermaidCode += `    ${node.id}(["${escapedLabel}"])\n`;
            mermaidCode += `    class ${node.id} default;\n`;
        });

        dagData.edges.forEach(edge => {
            mermaidCode += `    ${edge.from} --> ${edge.to};\n`;
        });
        mermaidContainer.textContent = mermaidCode;
        dagWrapper.appendChild(mermaidContainer);
        
        if (isDagModeEnabled) {
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'mt-6 flex gap-4 justify-center';
            // Added data-i18n and t() calls
            buttonContainer.innerHTML = `
                <button class="dag-yes-btn px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium shadow-md transition-all duration-200 flex items-center gap-2 transform hover:scale-105">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                    </svg>
                    <span data-i18n="btn_start_analysis">${t('btn_start_analysis', 'Start Analysing')}</span>
                </button>
                <button class="dag-no-btn px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium shadow-md transition-all duration-200 flex items-center gap-2 transform hover:scale-105">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
                    </svg>
                    <span data-i18n="btn_modify_dag">${t('btn_modify_dag', 'Modify DAG')}</span>
                </button>
            `;
            dagWrapper.appendChild(buttonContainer);
            buttonContainer.querySelector('.dag-yes-btn').addEventListener('click', handleYesClick);
            buttonContainer.querySelector('.dag-no-btn').addEventListener('click', handleNoClick);
    
            const exampleDiv = document.createElement('div');
            exampleDiv.className = 'mt-6 p-4 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-100 dark:border-blue-800/50 text-left w-full max-w-2xl mx-auto';
            // Added data-i18n and t() calls for Help Text
            exampleDiv.innerHTML = `
                <div class="w-full text-left" style="text-align: left !important;">
                    <div class="bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
                        <h4 class="font-semibold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2 text-left">
                            <span class="text-blue-500 text-lg" data-i18n="dag_tooltip_edit">${t('dag_tooltip_edit', '✨Edit the DAG by typing these commands directly in the chat box:')}</span>
                        </h4>
                        <div class="space-y-4 text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                            <div class="flex items-start gap-3">
                                <span class="text-blue-500 mt-0.5 flex-shrink-0">•</span>
                                <div class="text-left"><strong class="text-blue-600 dark:text-blue-400" data-i18n="dag_help_add">${t('dag_help_add', 'Add a node:')}</strong><span class="block mt-1" data-i18n="dag_help_add_desc">${t('dag_help_add_desc', 'Insert a new node between node X and node Y: [new content]')}</span></div>
                            </div>
                            <div class="flex items-start gap-3">
                                <span class="text-blue-500 mt-0.5 flex-shrink-0">•</span>
                                <div class="text-left"><strong class="text-blue-600 dark:text-blue-400" data-i18n="dag_help_mod">${t('dag_help_mod', 'Modify a node:')}</strong><span class="block mt-1" data-i18n="dag_help_mod_desc">${t('dag_help_mod_desc', 'Change node X to: [new full content]')}</span></div>
                            </div>
                            <div class="flex items-start gap-3">
                                <span class="text-blue-500 mt-0.5 flex-shrink-0">•</span>
                                <div class="text-left"><strong class="text-blue-600 dark:text-blue-400" data-i18n="dag_help_del">${t('dag_help_del', 'Delete a node:')}</strong><span class="block mt-1" data-i18n="dag_help_del_desc">${t('dag_help_del_desc', 'Delete node X')}</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            dagWrapper.appendChild(exampleDiv);
        }
        
        parentContainer.appendChild(dagWrapper);

        let tooltip = document.querySelector('.dag-tooltip');
        if (!tooltip) {
             tooltip = document.createElement('div');
             tooltip.className = 'dag-tooltip';
             tooltip.style.cssText = `
                position: absolute; background: linear-gradient(135deg, #6ab5d8ff 0%, #4763c1ff 50%, #2464f7ff 100%);
                color: white; padding: 6px 12px; border-radius: 8px; font-size: 13px; font-weight: 500;
                pointer-events: none; z-index: 9999; opacity: 0; transition: opacity 0.2s, transform 0.2s;
                white-space: pre-line; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); transform: translateY(5px);
            `;
            document.body.appendChild(tooltip);
        }

        const runMermaidRender = () => {
            mermaid.init(undefined, `#${mermaidId}`).then(() => {
                const svgElement = document.querySelector(`#${mermaidId} svg`);
                if (!svgElement) return;
                
                mermaidContainer.style.opacity = '1';
                svgElement.style.margin = '0 auto';
                svgElement.style.display = 'block';
                
                const nodeMap = {};
                dagData.nodes.forEach(node => { nodeMap[node.id] = node; });
                
                const defs = svgElement.querySelector('defs') || document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                if (!svgElement.querySelector('defs')) svgElement.insertBefore(defs, svgElement.firstChild);
                
                if (!defs.querySelector('#node-gradient')) {
                    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                    gradient.setAttribute('id', 'node-gradient');
                    gradient.setAttribute('x1', '0%');
                    gradient.setAttribute('y1', '0%');
                    gradient.setAttribute('x2', '100%');
                    gradient.setAttribute('y2', '100%');
                    gradient.innerHTML = '<stop offset="0%" stop-color="#bfdbfe"/><stop offset="100%" stop-color="#93c5fd"/>';
                    defs.appendChild(gradient);
                }
                
                const nodeGroups = svgElement.querySelectorAll('g[class*="node"]');
                nodeGroups.forEach(nodeGroup => {
                    const shape = nodeGroup.querySelector('rect, ellipse, circle');
                    const textElement = nodeGroup.querySelector('text');
                    
                    if (shape) {
                        nodeGroup.addEventListener('mouseenter', (e) => {
                            e.stopPropagation();
                            const nodeId = getNodeIdFromGroup(nodeGroup, nodeMap);
                            if (nodeId) {
                                const node = nodeMap[nodeId];
                                function wrapText(text, maxLength = 50) {
                                    if (!text || text.length <= maxLength) return text || '';
                                    const words = text.split(' ');
                                    let line = '';
                                    const result = [];
                                    for (const word of words) {
                                        if ((line + word).length > maxLength) {
                                            if (line) result.push(line.trim());
                                            line = word + ' ';
                                        } else {
                                            line += word + ' ';
                                        }
                                    }
                                    if (line) result.push(line.trim());
                                    return result.join('<br>');
                                }

                                // i18n applied to tooltip
                                tooltip.innerHTML = `
                                    <span class="font-bold text-blue-300 drop-shadow-md">${t('tooltip_content', 'Content:')}</span> ${wrapText(escapeHtml(node.label || 'No content available'), 55)}</span><br>
                                    <span class="font-bold text-emerald-400 drop-shadow-md">${t('tooltip_task_type', 'Task Type:')}</span> ${escapeHtml(node.tasktype || 'Unknown')}        
                                `.trim();
                                tooltip.style.left = `${e.pageX + 10}px`;
                                tooltip.style.top = `${e.pageY + 10}px`;
                                tooltip.style.opacity = '1';
                                tooltip.style.transform = 'translateY(0)';
                            }
                        });
                        
                        nodeGroup.addEventListener('mousemove', (e) => {
                            e.stopPropagation();
                            if (tooltip.style.opacity === '1') {
                                tooltip.style.left = `${e.pageX + 10}px`;
                                tooltip.style.top = `${e.pageY + 10}px`;
                            }
                        });
                        
                        nodeGroup.addEventListener('mouseleave', (e) => {
                            e.stopPropagation();
                            shape.style.filter = '';
                            shape.style.strokeWidth = '1.5';
                            shape.style.stroke = '#4f46e5';
                            shape.style.fill = 'url(#node-gradient)';
                            if (textElement) {
                                textElement.style.fill = '#1e293b';
                                textElement.style.fontWeight = '500';
                                textElement.style.fontSize = '13px';
                            }
                            tooltip.style.opacity = '0';
                            tooltip.style.transform = 'translateY(5px)';
                        });
                    }
                });
            }).catch(err => {
                console.error('DAG渲染失败:', err);
                mermaidContainer.style.opacity = '1'; 
                mermaidContainer.innerHTML = `<p class="text-red-500" data-i18n="error_render">${t('error_render', 'Render Failed')}</p>`;
            });
        };

        function getNodeIdFromGroup(nodeGroup, nodeMap) {
            const nodeGId = nodeGroup.id || '';
            let matchedNodeId = null;
            Object.keys(nodeMap).forEach(originalId => {
                if (nodeGId.includes(originalId) || nodeGroup.querySelector(`*[id*="${originalId}"]`)) {
                    matchedNodeId = originalId;
                }
            });
            return matchedNodeId;
        }

        if (document.body.contains(dagWrapper)) {
            runMermaidRender();
        } else {
            mermaidContainer._pendingRender = runMermaidRender;
        }
        
        scrollToBottom();
    }

    function appendContentToBlock(block, contentType, content) {
        let targetContainer;
        if (block.dataset.isThinking === 'true') {
            targetContainer = block.querySelector('.thinking-text');
        } else {
            targetContainer = block.querySelector('.result-content') || block.querySelector('.user-content');
        }

        if (!targetContainer) return;

        if (block.dataset.msgIndex !== undefined && activeSessionId && chatSessions[activeSessionId]) {
            const session = chatSessions[activeSessionId];
            const msgIndex = parseInt(block.dataset.msgIndex);
            
            if (session.messages && session.messages[msgIndex]) {
                const msg = session.messages[msgIndex];
                msg.contentType = contentType;
                
                if (contentType === 'text') {
                    msg.content = (msg.content || '') + content;
                } else if (contentType === 'dag') {
                    msg.content = content; 
                    if (content.id) msg.dagId = content.id;
                }
            }
        }

        if (contentType === 'text') {
            if (block.dataset.isThinking === 'true') {
                const cleanText = content.trim();
                if (cleanText) {
                    targetContainer.innerHTML += escapeHtml(cleanText);
                    if (/[。！？\n.!?]$/.test(cleanText)) {
                        targetContainer.innerHTML += '<br>';
                    } else {
                        targetContainer.innerHTML += ' ';
                    }
                }
            } else {
                const htmlContent = marked.parse(content);
                targetContainer.innerHTML += htmlContent;
            }
        }
        else if (contentType === 'dag') {
            renderDag(content, targetContainer);
        }

        scrollToBottom();
        if (contentType === 'dag' || contentType === 'code') {
            setTimeout(() => scrollToBottom(), 300);
        }
    }

    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function toggleThinkingBlock() {
        if (!currentThinkingBlock) return;
        const thinkingText = currentThinkingBlock.querySelector('.thinking-text');
        const collapseBtn = currentThinkingBlock.querySelector('.collapse-thinking-btn');
        if (thinkingText && collapseBtn) {
            const block = thinkingText.closest('.message-block');
            const isCollapsed = thinkingText.querySelector('.text-blue-600') !== null;
            
            if (isCollapsed) {
                const summary = thinkingText.querySelector('.text-blue-600');
                if (summary && summary.dataset.originalContent) {
                    thinkingText.innerHTML = summary.dataset.originalContent;
                    collapseBtn.innerHTML = `
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 w-4"><path stroke-linecap="round" stroke-linejoin="round" d="M4.5 15.75l7.5-7.5 7.5 7.5" /></svg>
                    `;
                }
            } else {
                toggleThinkingBlockState(block, true);
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
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
    }

    function collectAiResponseContent(resultBlock) {
        const contentContainer = resultBlock.querySelector('.result-content');
        if (!contentContainer) return '（No content）';
        let text = contentContainer.innerText || contentContainer.textContent;
        const now = new Date();
        const header = `=== AI Analysis Report ===\nTime: ${now.toLocaleString()}\n\n`;
        return header + text.trim();
    }

    function sendMessageWithSocket() {
        const message = chatInput.textContent.trim();
        if ((!message && !isDagReviewMode) || isProcessing) return;
        
        if (!activeSessionId) {
            createNewConversation();
        }

        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        currentConversationContainer = null;
        isProcessing = true;

        sendBtn.disabled = true;
        sendBtn.innerHTML = SEND_BTN_SPINNING;
        
        chatInput.textContent = '';
        adjustInputHeight();
        
        const cancelButton = document.querySelector('.cancel-modification-btn');
        if (cancelButton && cancelButton.parentNode) {
            cancelButton.parentNode.removeChild(cancelButton);
        }
        
        if (isDagReviewMode) {
            createMessageBlock('user', false, `${message}`);
        } else {
            createMessageBlock('user', false, message);
            const currentSession = chatSessions[activeSessionId];
            if (currentSession && currentSession.messages && currentSession.messages.length === 1 && !isDagReviewMode) {
                updateCurrentSidebarTitle(message);
            }
        }
        
        createMessageBlock('ai', true);
        
        const selectedModel = getSelectedModel();
        const payload = {
            model: selectedModel,
            expert_mode: isDagModeEnabled
        };

        if (isInteractiveModeEnabled) {
            payload.mode = "interact";
        }

        if (isDagReviewMode) {
            payload.is_dag_modification = "true";
            payload.dag_id = currentDagId || '';
            payload.modifications = message;
            payload.dag_confirm = "no";
        } else {
            payload.message = message;
        }

        socket.emit('chat_request', payload);
        
        if (isDagReviewMode) {
            isDagReviewMode = false;
            chatInput.setAttribute('placeholder', t('input_placeholder', 'Type your message...'));
        }
    }

    function resetSendButton() {
        if (!isProcessing) return; 
        isProcessing = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = SEND_BTN_NORMAL;
    }

    function adjustInputHeight() {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    }

    function sendMessage() {
        sendMessageWithSocket();
    }

    sendBtn.addEventListener('click', sendMessage);
    
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    chatInput.addEventListener('input', adjustInputHeight);
    adjustInputHeight();

    document.addEventListener('click', function(e) {
        if (e.target.closest('.collapse-thinking-btn')) {
            e.preventDefault();
            const block = e.target.closest('.message-block');
            currentThinkingBlock = block; 
            toggleThinkingBlock();
            return;
        }

        if (e.target.classList.contains('js-copy')) {
            const codeBlock = e.target.closest('.js-code-block');
            const codeElement = codeBlock.querySelector('code');
            const textToCopy = codeElement.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = e.target.textContent;
                e.target.textContent = t('btn_copied', 'Copied!');
                e.target.classList.add('success');
                setTimeout(() => {
                    e.target.textContent = originalText;
                    e.target.classList.remove('success');
                }, 2000);
            });
        }

        if (e.target.closest('.download-btn')) {
            e.preventDefault();
            const resultBlock = e.target.closest('.message-block');
            const content = collectAiResponseContent(resultBlock);
            if (content) {
                const now = new Date();
                const timestamp = now.toISOString().replace(/[:.]/g, '-');
                const filename = `ai-response-${timestamp}.txt`;
                downloadTextAsFile(content, filename);
            }
        }
    });

    const newConversationBtn = document.getElementById('btn-new-conversation');
    
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', function(e) {
            e.preventDefault(); 
            createNewConversation();
        });
    }

    function switchSession(sessionId) {
        if (activeSessionId === sessionId) return;

        if (activeSessionId && chatSessions[activeSessionId]) {
            const currentWrapper = chatSessions[activeSessionId].container;
            if (currentWrapper.parentNode === chatContainer) {
                chatContainer.removeChild(currentWrapper);
            }
        }

        activeSessionId = sessionId;
        const newSession = chatSessions[sessionId];

        chatContainer.appendChild(newSession.container);

        requestAnimationFrame(() => {
            const pendingMermaids = newSession.container.querySelectorAll('.mermaid');
            pendingMermaids.forEach(el => {
                if (el._pendingRender && typeof el._pendingRender === 'function') {
                    el._pendingRender(); 
                    delete el._pendingRender; 
                }
            });
        });

        currentThinkingBlock = null;
        currentResultBlock = null;
        currentMessageId = null;
        isProcessing = false; 
        currentConversationContainer = null; 
        isDagReviewMode = false;
        
        sendBtn.disabled = false;
        sendBtn.innerHTML = SEND_BTN_NORMAL;
        
        scrollToBottom();
        updateSidebarHighlight(sessionId);
        
        saveState(); 
    }

    function createNewConversation() {
        const sessionId = 'session-' + Date.now();
        const sessionDiv = document.createElement('div');
        sessionDiv.className = 'session-wrapper w-full h-full flex flex-col justify-end'; 
        sessionDiv.id = sessionId;
        
        // i18n for default title
        const defaultTitle = t('new_chat_default_title', 'New Conversation');

        chatSessions[sessionId] = {
            id: sessionId,
            container: sessionDiv,
            timestamp: new Date(),
            title: defaultTitle,
            messages: [] 
        };

        addNewSidebarItem(sessionId, defaultTitle);
        switchSession(sessionId);
        
        chatInput.textContent = '';
        document.getElementById('chat-input').focus();
        
        saveState();
    }

    function updateSidebarHighlight(activeId) {
        const allLinks = document.querySelectorAll('#chatAside ul a');
        allLinks.forEach(link => {
            link.classList.remove('active', 'bg-blue-100', 'dark:bg-blue-950');
            if (link.dataset.sessionId === activeId) {
                link.classList.add('active', 'bg-blue-100', 'dark:bg-blue-950');
            }
        });
    }

    function addNewSidebarItem(sessionId, customTitle = null) {
        if (!sessionId) return; 

        const newTitle = customTitle || t('new_chat_default_title', 'New Conversation') + " " + new Date().toLocaleTimeString();
        
        const li = document.createElement('li');
        li.className = 'group flex items-center justify-stretch max-w-full relative';
        
        li.innerHTML = `
            <a href="#" data-session-id="${sessionId}" class="flex gap-2 items-center px-4 py-3 w-full rounded-md transition-all">
                <div class="flex-shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="h-4 text-slate-600 dark:text-slate-200">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 0 1-.825-.242m9.345-8.334a2.126 2.126 0 0 0-.476-.095 48.64 48.64 0 0 0-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0 0 11.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
                    </svg>
                </div>
                <p class="text-sm text-slate-500 dark:text-slate-300 capitalize truncate conversation-title">${newTitle}</p>
            </a>
        `;

        if (conversationList) {
            if (!conversationList.querySelector(`a[data-session-id="${sessionId}"]`)) {
                conversationList.insertBefore(li, conversationList.firstChild);
            }
        }
        
        const anchor = li.querySelector('a');
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const clickedId = this.dataset.sessionId;
            switchSession(clickedId);
        });
    }

    function updateCurrentSidebarTitle(firstMessage) {
        const activeLink = document.querySelector(`#chatAside a[data-session-id="${activeSessionId}"]`);
        
        if (activeLink) {
            const titleEl = activeLink.querySelector('.conversation-title');
            if (titleEl) {
                const titleText = firstMessage.length > 20 ? firstMessage.substring(0, 20) + '...' : firstMessage;
                titleEl.textContent = titleText;
                
                if(chatSessions[activeSessionId]) {
                    chatSessions[activeSessionId].title = titleText;
                    saveState(); 
                }
            }
        }
    }

    function loadState() {
        const savedData = sessionStorage.getItem(STORAGE_KEY);
        const savedActiveId = sessionStorage.getItem(ACTIVE_SESSION_KEY);

        if (savedData) {
            try {
                const parsedSessions = JSON.parse(savedData);
                const sortedIds = Object.keys(parsedSessions).sort((a,b) => 
                    new Date(parsedSessions[a].timestamp) - new Date(parsedSessions[b].timestamp)
                );

                for (const id of sortedIds) {
                    const data = parsedSessions[id];
                    const sessionDiv = document.createElement('div');
                    sessionDiv.className = 'session-wrapper w-full h-full flex flex-col justify-end';
                    sessionDiv.id = id;

                    chatSessions[id] = {
                        id: id,
                        container: sessionDiv,
                        timestamp: new Date(data.timestamp),
                        title: data.title,
                        messages: data.messages || []
                    };
                    
                    addNewSidebarItem(id, data.title);

                    activeSessionId = id; 
                    currentConversationContainer = null; 

                    (data.messages || []).forEach(msg => {
                        const block = createMessageBlock(msg.sender, msg.isThinking, msg.content, false);
                        
                        if (msg.contentType === 'dag' && msg.content) {
                             const target = block.querySelector('.result-content');
                             if (target) {
                                 target.innerHTML = ''; 
                                 renderDag(msg.content, target);
                                 if (msg.dagId) block.dataset.dagId = msg.dagId;
                             }
                        }
                    });
                }

                if (savedActiveId && chatSessions[savedActiveId]) {
                    switchSession(savedActiveId);
                } else if (sortedIds.length > 0) {
                    switchSession(sortedIds[sortedIds.length-1]);
                }
                return true; 
            } catch (e) {
                console.error('Failed to load session state:', e);
                sessionStorage.removeItem(STORAGE_KEY);
                return false;
            }
        }
        return false;
    }

    loadState();
    console.log('聊天界面初始化完成 (With i18n & Persistence)');
});