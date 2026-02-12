# 中英文切换功能说明

## 已完成的配置

### 1. 导航栏语言切换器
在 `docusaurus.config.js` 中添加了语言切换下拉菜单：
```javascript
{
  type: 'localeDropdown',
  position: 'right',
}
```

### 2. 翻译文件结构

#### 中文翻译文件 (zh-CN)
- `i18n/zh-CN/code.json` - 首页翻译
- `i18n/zh-CN/docusaurus-theme-classic/navbar.json` - 导航栏翻译
- `i18n/zh-CN/docusaurus-theme-classic/footer.json` - 页脚翻译
- `i18n/zh-CN/docusaurus-plugin-content-docs/current.json` - 侧边栏分类翻译
- `i18n/zh-CN/docusaurus-plugin-content-docs/current/tutorial-basics/_category_.json` - "从这里开始"分类
- `i18n/zh-CN/docusaurus-plugin-content-docs/current/tutorial-extras/_category_.json` - "上手案例"分类
- `i18n/zh-CN/docusaurus-plugin-content-docs/current/tutorial-algorithm/_category_.json` - "专用算子"分类

#### 英文翻译文件 (en)
- `i18n/en/code.json` - 首页翻译
- `i18n/en/docusaurus-theme-classic/navbar.json` - 导航栏翻译
- `i18n/en/docusaurus-theme-classic/footer.json` - 页脚翻译
- `i18n/en/docusaurus-plugin-content-docs/current.json` - 侧边栏分类翻译
- `i18n/en/docusaurus-plugin-content-docs/current/intro.md` - 简介页面英文版
- `i18n/en/docusaurus-plugin-content-docs/current/tutorial-basics/_category_.json` - "Getting Started"分类
- `i18n/en/docusaurus-plugin-content-docs/current/tutorial-extras/_category_.json` - "Tutorials"分类
- `i18n/en/docusaurus-plugin-content-docs/current/tutorial-algorithm/_category_.json` - "Algorithms"分类

## 功能说明

### 如何使用
1. 访问网站后，在导航栏右上角会看到语言切换下拉菜单
2. 点击下拉菜单可以选择"简体中文"或"English"
3. 选择后页面会自动切换到对应语言

### 当前翻译覆盖范围
- ✅ 首页 (index.js) - 完全支持中英文切换
- ✅ 导航栏 - 完全支持中英文切换
- ✅ 页脚 - 完全支持中英文切换
- ✅ 侧边栏分类标签 - 完全支持中英文切换
  - "基本信息" / "Basic Information"
  - "从这里开始" / "Getting Started"
  - "上手案例" / "Tutorials"
  - "专用算子" / "Algorithms"
- ✅ 简介页面 (intro.md) - 已有英文版本

### 如何添加更多页面的翻译

如果需要为其他文档页面添加英文翻译，请按以下步骤操作：

1. 在 `i18n/en/docusaurus-plugin-content-docs/current/` 目录下创建对应的 Markdown 文件
2. 文件路径和名称需要与中文版本保持一致
3. 例如：
   - 中文：`docs/tutorial-basics/create-a-page.md`
   - 英文：`i18n/en/docusaurus-plugin-content-docs/current/tutorial-basics/create-a-page.md`

## 测试方法

运行开发服务器：
```bash
cd AAG_3/AAG/docs-site
npm start
```

然后在浏览器中访问 http://localhost:3000，测试语言切换功能。

## 注意事项

1. 默认语言是简体中文 (zh-CN)
2. 如果某个页面没有提供英文翻译，切换到英文时会显示中文内容
3. 所有翻译文件都使用 JSON 格式，需要保持正确的 JSON 语法
4. 侧边栏的分类标签已经完全支持中英文切换
5. 具体的文档页面内容需要单独创建对应的 Markdown 文件进行翻译
