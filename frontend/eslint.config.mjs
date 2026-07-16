import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypeScript from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTypeScript,
  {
    files: ["src/app/page.tsx"],
    // Existing dashboard lifecycle debt is bounded separately; all other
    // ESLint errors remain blocking and the warning budget prevents drift.
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "react-hooks/set-state-in-effect": "off",
    },
  },
  globalIgnores([".next/**", "test-results/**", "tsconfig.tsbuildinfo"]),
]);
