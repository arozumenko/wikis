import { source } from '@/lib/source';
import {
  DocsPage,
  DocsBody,
  DocsTitle,
  DocsDescription,
} from 'fumadocs-ui/page';
import { getMDXComponents } from '@/mdx-components';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import type { MDXContent } from 'mdx/types';
import type { TOCItemType } from 'fumadocs-core/toc';
import type { InferPageType } from 'fumadocs-core/source';

// fumadocs-mdx enriches page.data with body and toc at runtime;
// declare the extended shape so TypeScript is happy.
type DocsPageData = InferPageType<typeof source>['data'] & {
  body: MDXContent;
  toc: TOCItemType[];
};

interface Props {
  params: Promise<{ slug?: string[] }>;
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) return {};
  return {
    title: page.data.title,
    description: page.data.description,
  };
}

export default async function Page({ params }: Props) {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) notFound();

  const data = page.data as DocsPageData;
  const MDX = data.body;

  return (
    <DocsPage toc={data.toc} full={false}>
      <DocsTitle>{data.title}</DocsTitle>
      <DocsDescription>{data.description}</DocsDescription>
      <DocsBody>
        <MDX components={getMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}
