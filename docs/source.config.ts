import { defineDocs, defineConfig } from 'fumadocs-mdx/config';
import { rehypeCodeDefaultOptions } from 'fumadocs-core/mdx-plugins';
import { visit } from 'unist-util-visit';
import type { Root, Element } from 'hast';
import type { Plugin } from 'unified';

// Rewrite all /guide/* links → /docs/* so Mintlify-style hrefs work in Fumadocs
const rehypeRewriteGuideLinks: Plugin<[], Root> = () => (tree) => {
  visit(tree, 'element', (node: Element) => {
    if (
      node.tagName === 'a' &&
      node.properties?.href &&
      typeof node.properties.href === 'string' &&
      node.properties.href.startsWith('/guide/')
    ) {
      node.properties.href = node.properties.href.replace(/^\/guide\//, '/docs/');
    }
  });
};

// Rename raw JSX <img src="/images/..."> nodes → <ThemedImage> so the
// components map can intercept them (raw JSX bypasses the 'img' override).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const remarkThemedImages: Plugin<[], any> = () => (tree) => {
  visit(tree, (node: any) => {
    if (
      (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') &&
      node.name === 'img'
    ) {
      const srcAttr = node.attributes?.find(
        (a: any) => a.type === 'mdxJsxAttribute' && a.name === 'src'
      );
      const src = typeof srcAttr?.value === 'string' ? srcAttr.value : srcAttr?.value?.value;
      if (src && typeof src === 'string' && src.startsWith('/images/')) {
        node.name = 'ThemedImage';
      }
    }
  });
};

export const { docs, meta } = defineDocs({
  dir: 'guide',
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: (v) => [remarkThemedImages, ...v],
    rehypePlugins: (v) => [...v, rehypeRewriteGuideLinks],
    rehypeCodeOptions: {
      ...rehypeCodeDefaultOptions,
      fallbackLanguage: 'text',
    },
  },
});
