import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "QUANTEX Trading System",
  description: "Autonomous AI Multi-Agent Trading System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{
        margin: 0,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        backgroundColor: "#0a0a0f",
        color: "#e0e0e0",
      }}>
        {children}
      </body>
    </html>
  );
}
