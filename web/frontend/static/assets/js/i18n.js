/**
 * i18n.js - 通用国际化/语言切换脚本
 * 集成了本地缓存优化以解决 FOUTC (Flash of Untranslated Content) 问题
 */

// 初始化全局变量，确保其他脚本（如 chat.js）能访问
window.LANG = {};

// --- 辅助函数：显示页面 ---
// 配合 CSS 使用：body.lang-loading { opacity: 0; } body.lang-ready { opacity: 1; }
function revealPage() {
    document.body.classList.remove('lang-loading');
    document.body.classList.add('lang-ready');
}

// --- 核心逻辑：应用语言数据 ---
function applyLanguageData(data, lang) {
    // 1. 设置 JS 全局变量
    window.LANG = data.js_vars || {};

    // 2. 映射 HTML 静态内容
    const htmlMap = data.html_map || {};
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
            // 特殊处理 ContentEditable DIV (如聊天输入框)
            else if (el.id === 'chat-input') {
                el.setAttribute('placeholder', text); 
            }
            // 普通文本
            else {
                // 默认使用 innerText 防止 XSS，除非明确需要 HTML
                el.innerText = text;
            }
        }
    }
    
    // 3. 保存 html_map 到全局变量
    window.HTML_MAP = htmlMap;
    
    // 4. 处理带有 data-placeholder-id 属性的输入框
    document.querySelectorAll('[data-placeholder-id]').forEach(input => {
        const placeholderId = input.getAttribute('data-placeholder-id');
        if (htmlMap[placeholderId]) {
            input.placeholder = htmlMap[placeholderId];
        }
    });
    
    // 5. 触发自定义事件，通知其他脚本语言已更新
    document.dispatchEvent(new Event('languageLoaded'));
}

// --- 核心逻辑：获取语言数据 (网络请求) ---
async function fetchLanguageData(lang, isBackgroundUpdate = false) {
    try {
        const response = await fetch(`/static/language/${lang}.json`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        
        // 更新缓存
        localStorage.setItem(`i18n_cache_${lang}`, JSON.stringify(data));

        // 应用数据
        applyLanguageData(data, lang);

        if (!isBackgroundUpdate) {
            // 如果不是后台更新（即首次加载），需要显示页面
            revealPage();
            console.log(`Loaded ${lang} from network`);
        } else {
            console.log(`Updated ${lang} cache in background`);
        }

    } catch (error) {
        console.error('Fetch language error:', error);
        // 如果出错且不是后台更新，也要显示页面，防止白屏
        if (!isBackgroundUpdate) revealPage();
    }
}

// --- 主函数：加载语言 ---
async function loadLanguage(lang) {
    try {
        // 1. 保存设置
        localStorage.setItem('app_lang', lang);
        document.documentElement.lang = lang;

        // 2. 更新切换按钮文字
        const btn = document.getElementById('lang-toggle-btn');
        if (btn) {
            btn.textContent = (lang === 'zh-CN') ? 'English' : '简体中文';
        }

        // 3. 缓存策略：优先使用缓存渲染，后台静默更新
        const cachedDataStr = localStorage.getItem(`i18n_cache_${lang}`);
        
        if (cachedDataStr) {
            try {
                const cachedData = JSON.parse(cachedDataStr);
                // 命中缓存：立即渲染并显示页面
                applyLanguageData(cachedData, lang);
                revealPage();
                console.log(`Loaded ${lang} from cache (Instant render)`);
                
                // 后台发起请求更新缓存（保持数据最新）
                fetchLanguageData(lang, true);
                return;
            } catch (e) {
                console.warn('Cache parse error, falling back to network', e);
            }
        }

        // 4. 未命中缓存：等待网络请求
        await fetchLanguageData(lang, false);

    } catch (error) {
        console.error('Failed to load language flow:', error);
        revealPage();
    }
}

// 切换语言函数 (供按钮点击调用)
function toggleLanguage() {
    const current = localStorage.getItem('app_lang') || 'zh-CN';
    const next = current === 'zh-CN' ? 'en-US' : 'zh-CN';
    return loadLanguage(next);
}

// 绑定语言切换按钮的函数
let buttonBound = false;
let clickHandler = null;

function bindLanguageToggleButton() {
    const btn = document.getElementById('lang-toggle-btn');
    if (!btn) {
        // console.log('Language toggle button not found yet');
        return false;
    }
    
    if (btn.hasAttribute('data-i18n-bound')) {
        return true;
    }
    
    if (!clickHandler) {
        clickHandler = function(e) {
            if (e) {
                e.stopPropagation();
                e.preventDefault();
            }
            console.log('Language toggle button clicked');
            
            if (typeof toggleLanguage === 'function') {
                toggleLanguage().catch(err => console.error(err));
            } else {
                // 兜底方案
                const current = localStorage.getItem('app_lang') || 'zh-CN';
                const next = current === 'zh-CN' ? 'en-US' : 'zh-CN';
                localStorage.setItem('app_lang', next);
                location.reload();
            }
        };
    }
    
    // 绑定事件
    btn.addEventListener('click', clickHandler, { capture: false, passive: false });
    
    // 同时绑定父元素 LI 以增加点击区域
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
    return true;
}

// 页面加载初始化
function initI18n() {
    // 1. 读取存储的语言，默认为中文
    const savedLang = localStorage.getItem('app_lang') || 'zh-CN';
    
    // 2. 加载语言 (如果 body 有 lang-loading 类，此时会处理显示逻辑)
    loadLanguage(savedLang);
    
    // 3. 尝试绑定按钮 (重试机制)
    const tryBind = () => {
        if (!bindLanguageToggleButton()) {
            setTimeout(tryBind, 50);
        }
    };
    tryBind();
    setTimeout(tryBind, 300);
}

// 根据文档状态启动
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initI18n);
} else {
    initI18n();
}

// 额外保障：监听 DOM 变化以绑定后来出现的按钮
if (typeof MutationObserver !== 'undefined') {
    const observer = new MutationObserver(() => {
        const btn = document.getElementById('lang-toggle-btn');
        if (btn && !btn.hasAttribute('data-i18n-bound')) {
            bindLanguageToggleButton();
        }
    });
    
    const observeTarget = document.body || document.documentElement;
    observer.observe(observeTarget, {
        childList: true,
        subtree: true
    });
}