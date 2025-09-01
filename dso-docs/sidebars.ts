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
      label: 'Getting Started',
      items: [
        'core/getting_started',
        'core/setup',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core/concept',
        'core/architecture',
        'core/tokens',
        'core/training',
        'core/constraints',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      items: [
        'core/mimo',
        'core/advanced',
      ],
    },
    {
      type: 'category',
      label: 'Rules',
      items: [
        'rules/README',
        'rules/branching_model',
        'rules/code_style',
        'rules/docs_rules',
        'rules/git_rules',
        'rules/pr_review_rules',
        'rules/project_structure',
        'rules/security_rules',
        'rules/testing_rules',
      ],
    },
  ],
};

export default sidebars;
