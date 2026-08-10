import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement scrollIntoView (real browsers do — verified live
// in the actual Browser pane during Phase 10 manual testing) — polyfill it
// as a no-op so components that call it don't need environment-detection
// code of their own just to satisfy the test environment.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
