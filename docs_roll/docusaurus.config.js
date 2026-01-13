// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const { themes } = require('prism-react-renderer');
const lightTheme = themes.github;
const darkTheme = themes.dracula;

const sidebars = require('./sidebars.js');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ROLL',
  // tagline: 'Dinosaurs are cool',
  favicon: 'https://img.alicdn.com/imgextra/i4/O1CN01bo6EZl2192CAIjFwE_!!6000000006941-2-tps-465-367.png',

  // Set the production url of your site here
  url: 'https://alibaba.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/ROLL/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'alibaba', // Usually your GitHub org/user name.
  projectName: 'ROLL', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      'zh-CN': {
        label: '简体中文',
        direction: 'ltr',
      },
    },
  },

  customFields: {
    fullSidebar: sidebars.tutorialSidebar,
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/alibaba/ROLL/tree/main/docs_roll/',
          showLastUpdateTime: true
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: 'G-D6R4GXHVFP',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themes: [
    [
      // @ts-ignore
      require.resolve("@easyops-cn/docusaurus-search-local"),
      /** @type {import("@easyops-cn/docusaurus-search-local").PluginOptions} */
      // @ts-ignore
      ({
        hashed: true,
        indexBlog: false,
        // For Docs usingChinese, it is recomended to set:
        language: ["en"],
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'ROLL',
        logo: {
          alt: 'ROLL Logo',
          src: 'https://img.alicdn.com/imgextra/i3/O1CN016Mlxas1MHNA3NEbZ0_!!6000000001409-2-tps-465-367.png',
        },
        items: [
          { to: '/ROLL', label: 'Home', position: 'right' },
          { to: '/ROLL/#core', label: 'Core Algorithms', position: 'right' },
          { to: '/ROLL/#research', label: 'Research Community', position: 'right' },
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'right',
            label: 'API Docs',
          },
          {
            href: 'https://github.com/alibaba/ROLL',
            label: 'GitHub',
            position: 'right',
          },
          {
            type: 'search',
            position: 'right', // 确保位置在右侧
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Examples',
            items: [
              {
                label: 'ROLL单机实践手册',
                to: '/docs/Getting%20Started/Quick%20Start/single_node_quick_start',
              },
              {
                label: '配置指南',
                to: '/docs/User%20Guides/Configuration/config_guide',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/alibaba/ROLL',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Alibaba.`,
      },
      prism: {
        theme: lightTheme,
        darkTheme: darkTheme,
      },
      colorMode: {
        defaultMode: 'dark',
      },
    }),
};

module.exports = config;
