import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "Deep Symbolic Optimization",
  tagline: "AI-powered discovery of interpretable mathematical expressions",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: "https://your-docusaurus-site.example.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "facebook", // Usually your GitHub org/user name.
  projectName: "docusaurus", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Serve the docs at the site root (i.e. '/').
          routeBasePath: "/",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/sheydHD/deep-symbolic-optimization",
        },
        // Disable the blog since you don't need default sample blog content
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ["@docusaurus/theme-mermaid"],
  
  markdown: {
    mermaid: true,
  },

  themeConfig: {
    // Replace with your project's social card
    image: "img/docusaurus-social-card.jpg",
    navbar: {
      title: "Deep Symbolic Optimization",
      logo: {
        alt: "DSO Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          to: "/regression/overview",
          label: "Regression",
          position: "left",
        },
        {
          to: "/examples/basic-regression",
          label: "Examples",
          position: "left",
        },
        {
          to: "/api/core-api",
          label: "API",
          position: "left",
        },
        {
          href: "https://github.com/sheydHD/deep-symbolic-optimization",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Documentation",
          items: [
            {
              label: "Installation",
              to: "/installation/requirements",
            },
            {
              label: "Regression",
              to: "/regression/overview",
            },
            {
              label: "Examples",
              to: "/examples/basic-regression",
            },
            {
              label: "API Reference",
              to: "/api/core-api",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "GitHub Repository",
              href: "https://github.com/sheydHD/deep-symbolic-optimization",
            },
            {
              label: "Research Papers",
              href: "https://github.com/sheydHD/deep-symbolic-optimization#publications",
            },
          ],
        },
        {
          title: "Applications",
          items: [
            {
              label: "Scientific Computing",
              to: "/regression/overview",
            },
            {
              label: "Engineering",
              to: "/examples/basic-regression",
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Deep Symbolic Optimization. Open source framework for symbolic regression.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
