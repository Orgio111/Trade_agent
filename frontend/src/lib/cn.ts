/**
 * Utility to merge class names, filtering out falsy values.
 * Lightweight alternative to clsx/twMerge for this project.
 */
export function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
