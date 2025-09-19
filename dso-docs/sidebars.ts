import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Installation',
      items: [
        'installation/requirements',
        'installation/setup',
        'installation/quickstart',
      ],
    },
    {
      type: 'category',
      label: 'Regression',
      items: [
        'regression/overview',
        'regression/symbolic-regression',
        'regression/single-output',
        'regression/multi-output',
        'regression/configuration',
        'regression/optimization',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/core-api',
        'core/tokens',
        'core/constraints',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/basic-regression',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'core/architecture',
        'core/training',
        'core/advanced',
      ],
    },
    'summary',
  ],
};

export default sidebars;
