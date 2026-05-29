/* ─── Background atmospheric effects ────────────────────────────────────── */
export default function BackgroundEffects() {
  return (
    <>
      {/* Grid overlay */}
      <div className="fixed inset-0 hud-grid-bg pointer-events-none z-0" />

      {/* Glow orbs */}
      <div className="glow-orb-1" />
      <div className="glow-orb-2" />

      {/* Top accent line */}
      <div className="fixed top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-hud-indigo/50 to-transparent pointer-events-none z-50" />
    </>
  );
}
