import { RootProvider } from 'fumadocs-ui/provider';
import { Inter, Playfair_Display, JetBrains_Mono } from 'next/font/google';
import type { ReactNode } from 'react';
import 'fumadocs-ui/style.css';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const playfair = Playfair_Display({ subsets: ['latin'], variable: '--font-playfair' });
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' });

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${playfair.variable} ${jetbrains.variable}`}
      suppressHydrationWarning
    >
      <body className={`${inter.className} flex min-h-screen flex-col`}>
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
