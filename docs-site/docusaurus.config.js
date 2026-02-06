// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AAG Documentation',
  tagline: 'AAG 用户手册',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // GitHub Pages（用你自己的仓库）
  url: 'https://neugjq.github.io',
  baseUrl: '/AAG/',

  organizationName: 'neugjq',
  projectName: 'AAG',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'zh-CN',
    locales: ['zh-CN'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: 'docs', // 文档路径
          sidebarPath: './sidebars.js',

          // 👉 指向你要提交文档的“上游仓库”
          editUrl: 'https://github.com/superccy/AAG/tree/main/docs-site/',
        },

        // ❌ 不需要 Blog，关掉更干净
        blog: false,

        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'AAG中文文档',
      items: [
      { to: '/docs/intro', label: '指南', position: 'left' },
      { to: '/docs/intro', label: 'API文档', position: 'left' },
      { to: '/docs/intro', label: '开发者指南', position: 'left' },
      { href: 'https://github.com/superccy/AAG', label: 'GitHub', position: 'right' },
    ],
    },
    
    // 添加搜索栏
    algolia: undefined, // 如果需要搜索功能，可以配置 Algolia

    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} AAG`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
