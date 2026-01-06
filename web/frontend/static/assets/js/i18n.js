/**
 * i18n.js - 通用国际化/语言切换脚本
 */

// 初始化全局变量，确保其他脚本（如 chat.js）能访问
window.LANG = {};

// 核心加载语言函数
async function loadLanguage(lang) {
    try {
        // 1. 保存设置到本地存储
        localStorage.setItem('app_lang', lang);
        document.documentElement.lang = lang; // 修改 html 标签的 lang 属性

        // --- 新增代码开始：更新切换按钮的文字 ---
        // 逻辑：按钮显示当前语言，English对应英文界面，简体中文对应中文界面
        const btn = document.getElementById('lang-toggle-btn');
        if (btn) {
            if (lang === 'zh-CN') {
                btn.textContent = '简体中文';
            } else {
                btn.textContent = 'English';
            }
        }
        // --- 新增代码结束 ---

        // 请求语言文件
        // 假设 JSON 文件都在 /static/language/ 目录下
        const response = await fetch(`/static/language/${lang}.json`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        
        // 2. 设置 JS 全局变量 (给 chat.js 等逻辑脚本使用)
        window.LANG = data.js_vars;

        // 3. 映射 HTML 静态内容 (根据 ID 自动替换)
        const htmlMap = data.html_map;
        for (const [id, text] of Object.entries(htmlMap)) {
            const el = document.getElementById(id);
            if (el) {
                // 特殊处理 input placeholder
                if (el.tagName === 'INPUT' && el.hasAttribute('placeholder')) {
                    el.placeholder = text;
                } 
                // 特殊处理 textarea placeholder
                else if (el.tagName === 'TEXTAREA' && el.hasAttribute('placeholder')) {
                    el.placeholder = text;
                }
                // 特殊处理 contentEditable div (模拟 placeholder)
                else if (el.id === 'chat-input') {
                        el.setAttribute('placeholder', text); 
                }
                else {
                    // 普通文本替换
                    el.innerText = text;
                }
            }
        }
        
        // 4. 保存 html_map 到全局变量，供其他函数使用
        window.HTML_MAP = htmlMap;
        
        // 5. 处理带有 data-placeholder-id 属性的输入框的 placeholder
        document.querySelectorAll('[data-placeholder-id]').forEach(input => {
            const placeholderId = input.getAttribute('data-placeholder-id');
            if (htmlMap[placeholderId]) {
                input.placeholder = htmlMap[placeholderId];
            }
        });
        
        // 6. 触发自定义事件，通知其他脚本语言已更新
        // 其他页面的特定 JS (如 dashboard.js) 可以监听这个事件来刷新它的动态内容
        document.dispatchEvent(new Event('languageLoaded'));
        console.log(`Language loaded: ${lang}`);
        
    } catch (error) {
        console.error('Failed to load language:', error);
    }
}

// 切换语言函数 (供按钮点击调用)
function toggleLanguage() {
    const current = localStorage.getItem('app_lang') || 'zh-CN';
    const next = current === 'zh-CN' ? 'en-US' : 'zh-CN';
    return loadLanguage(next);
}

// 绑定语言切换按钮的函数（使用标志避免重复绑定）
let buttonBound = false;
let clickHandler = null;

function bindLanguageToggleButton() {
    const btn = document.getElementById('lang-toggle-btn');
    if (!btn) {
        console.log('Language toggle button not found');
        return false;
    }
    
    // 如果已经绑定过，不再重复绑定
    if (btn.hasAttribute('data-i18n-bound')) {
        console.log('Language toggle button already bound');
        return true;
    }
    
    // 创建事件处理函数（只创建一次）
    if (!clickHandler) {
        clickHandler = function(e) {
            // 阻止事件冒泡，避免被其他元素拦截
            if (e) {
                e.stopPropagation();
                e.preventDefault();
            }
            console.log('Language toggle button clicked', new Date().getTime());
            
            // 获取按钮引用
            const clickedBtn = e ? e.target.closest('#lang-toggle-btn') || document.getElementById('lang-toggle-btn') : document.getElementById('lang-toggle-btn');
            
            // 调用切换语言函数
            if (typeof toggleLanguage === 'function') {
                try {
                    // 立即调用，不等待
                    const result = toggleLanguage();
                    // 如果返回Promise，处理它
                    if (result && typeof result.then === 'function') {
                        result.catch((error) => {
                            console.error('Error in toggleLanguage:', error);
                        });
                    }
                } catch (error) {
                    console.error('Error in toggleLanguage:', error);
                    // 备用方案：直接切换语言并刷新页面
                    const current = localStorage.getItem('app_lang') || 'zh-CN';
                    const next = current === 'zh-CN' ? 'en-US' : 'zh-CN';
                    localStorage.setItem('app_lang', next);
                    location.reload();
                }
            } else {
                console.error('toggleLanguage function not available');
                // 备用方案：直接切换语言并刷新页面
                const current = localStorage.getItem('app_lang') || 'zh-CN';
                const next = current === 'zh-CN' ? 'en-US' : 'zh-CN';
                localStorage.setItem('app_lang', next);
                location.reload();
            }
        };
    }
    
    // 绑定事件监听器（使用capture阶段确保能捕获到事件）
    // 同时绑定到按钮和li元素，确保能捕获到点击
    btn.addEventListener('click', clickHandler, { 
        capture: false,  // 在冒泡阶段处理
        passive: false   // 允许preventDefault
    });
    
    // 也绑定到父元素li，以防按钮被遮挡
    const liParent = btn.closest('li');
    if (liParent && !liParent.hasAttribute('data-i18n-bound')) {
        liParent.addEventListener('click', function(e) {
            if (e.target === btn || btn.contains(e.target)) {
                clickHandler(e);
            }
        }, { capture: false, passive: false });
        liParent.setAttribute('data-i18n-bound', 'true');
    }
    btn.setAttribute('data-i18n-bound', 'true');
    buttonBound = true;
    
    console.log('Language toggle button bound successfully');
    return true;
}

// 页面加载初始化
function initI18n() {
    // 1. 读取存储的语言，默认为中文
    const savedLang = localStorage.getItem('app_lang') || 'zh-CN';
    loadLanguage(savedLang);
    
    // 2. 尝试绑定切换按钮（多次重试确保成功）
    const tryBind = () => {
        if (!bindLanguageToggleButton()) {
            // 如果按钮还不存在，延迟重试
            setTimeout(tryBind, 50);
        }
    };
    
    // 立即尝试绑定
    tryBind();
    
    // 延迟重试，确保按钮已加载
    setTimeout(tryBind, 100);
    setTimeout(tryBind, 300);
    setTimeout(tryBind, 500);
}

// 根据文档状态决定如何初始化
if (document.readyState === 'loading') {
    // DOM还在加载中，等待DOMContentLoaded事件
    document.addEventListener('DOMContentLoaded', initI18n);
} else {
    // DOM已经加载完成，立即初始化
    initI18n();
}

// 额外保障：监听DOM变化，如果按钮后来才出现，也能绑定
if (typeof MutationObserver !== 'undefined') {
    const observer = new MutationObserver(() => {
        const btn = document.getElementById('lang-toggle-btn');
        if (btn && !btn.hasAttribute('data-i18n-bound')) {
            bindLanguageToggleButton();
        }
    });
    
    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    } else {
        document.addEventListener('DOMContentLoaded', () => {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    }
}