import defaultComponents from 'fumadocs-ui/mdx';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { Steps, Step } from 'fumadocs-ui/components/steps';
import { Tab as FumaTab, Tabs as FumaTabs } from 'fumadocs-ui/components/tabs';
import { Callout } from 'fumadocs-ui/components/callout';
import type { MDXComponents } from 'mdx/types';
import type { ReactNode, ComponentProps, ImgHTMLAttributes } from 'react';

// Render both light and dark images; CSS hides the inactive one.
// Dark variant is automatically derived: /images/foo.png → /images/foo-dark.png
function ThemedImage({ src, alt, style, ...rest }: ImgHTMLAttributes<HTMLImageElement>) {
  if (!src || typeof src !== 'string' || !src.startsWith('/images/')) {
    return <img src={src} alt={alt} style={style} {...rest} />;
  }
  const darkSrc = src.replace(/(\.[^.]+)$/, '-dark$1');
  const baseStyle = { borderRadius: '0.5rem', border: '1px solid var(--color-fd-border)', maxWidth: '100%', ...style };
  return (
    <>
      <img src={src} alt={alt} style={baseStyle} className="wikis-img-light" {...rest} />
      <img src={darkSrc} alt={alt} style={baseStyle} className="wikis-img-dark" {...rest} />
    </>
  );
}
import {
  BookOpen, Zap, Search, Plug, Network, Share2, Lock, Microscope,
  Key, Settings, FileText, Code2, GitBranch, HelpCircle, Info, Terminal,
  Globe, Shield, Database, Cpu, LayoutDashboard, RefreshCw,
} from 'lucide-react';

// Mintlify FontAwesome icon name → Lucide React icon
const ICON_MAP: Record<string, ReactNode> = {
  'bolt':             <Zap className="size-4" />,
  'book-open':        <BookOpen className="size-4" />,
  'magnifying-glass': <Search className="size-4" />,
  'plug':             <Plug className="size-4" />,
  'diagram-project':  <Network className="size-4" />,
  'share-nodes':      <Share2 className="size-4" />,
  'lock':             <Lock className="size-4" />,
  'microscope':       <Microscope className="size-4" />,
  'key':              <Key className="size-4" />,
  'settings':         <Settings className="size-4" />,
  'file-text':        <FileText className="size-4" />,
  'code':             <Code2 className="size-4" />,
  'git-branch':       <GitBranch className="size-4" />,
  'circle-question':  <HelpCircle className="size-4" />,
  'info':             <Info className="size-4" />,
  'terminal':         <Terminal className="size-4" />,
  'globe':            <Globe className="size-4" />,
  'shield':           <Shield className="size-4" />,
  'database':         <Database className="size-4" />,
  'cpu':              <Cpu className="size-4" />,
  'layout-dashboard': <LayoutDashboard className="size-4" />,
  'rotate':           <RefreshCw className="size-4" />,
};

// Mintlify <Card icon="string" href="/guide/xxx"> → Fumadocs Card with icon ReactNode
// Also rewrites /guide/xxx hrefs to /docs/xxx
function MintlifyCard({
  icon,
  href,
  children,
  ...rest
}: ComponentProps<typeof Card> & { icon?: string | ReactNode }) {
  const resolvedIcon =
    typeof icon === 'string' ? (ICON_MAP[icon] ?? undefined) : icon;
  const resolvedHref =
    typeof href === 'string' ? href.replace(/^\/guide\//, '/docs/') : href;
  return (
    <Card icon={resolvedIcon} href={resolvedHref} {...rest}>
      {children}
    </Card>
  );
}

// Mintlify's <Tab title="..."> maps to fumadocs <Tab value="...">
function Tab({
  title,
  value,
  children,
  ...rest
}: { title?: string; value?: string; children?: ReactNode } & Omit<
  ComponentProps<typeof FumaTab>,
  'value' | 'children'
>) {
  const resolvedValue = value ?? title ?? '';
  return (
    <FumaTab value={resolvedValue} {...rest}>
      {children}
    </FumaTab>
  );
}

// Mintlify's <Tabs> doesn't require items — derive them from children's title props
function Tabs({
  children,
  ...rest
}: { children?: ReactNode } & Omit<ComponentProps<typeof FumaTabs>, 'items' | 'children'>) {
  const childArray = Array.isArray(children) ? children : children ? [children] : [];
  const items: string[] = [];
  for (const child of childArray) {
    if (child && typeof child === 'object' && 'props' in child) {
      const label: string | undefined =
        (child as { props: Record<string, unknown> }).props['title'] as string | undefined ??
        (child as { props: Record<string, unknown> }).props['value'] as string | undefined;
      if (label) items.push(label);
    }
  }
  return (
    <FumaTabs items={items.length > 0 ? items : undefined} {...rest}>
      {children}
    </FumaTabs>
  );
}

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    img: ThemedImage,
    ThemedImage,
    Card: MintlifyCard,
    Cards,
    Steps,
    Step,
    Tab,
    Tabs,
    Callout,
    CardGroup: ({ cols, children }: { cols?: number; children: ReactNode }) => (
      <Cards>
        {children}
      </Cards>
    ),
    Note: ({ children }: { children: ReactNode }) => (
      <Callout type="info">{children}</Callout>
    ),
    Tip: ({ children }: { children: ReactNode }) => (
      <Callout type="info">{children}</Callout>
    ),
    Warning: ({ children }: { children: ReactNode }) => (
      <Callout type="warn">{children}</Callout>
    ),
    ...components,
  };
}

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return getMDXComponents(components);
}
