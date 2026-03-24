import { describe, expect, it } from "vitest";

import { formatBytes, formatDurationSeconds } from "../format";

describe("format", () => {
  it("formats bytes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1024 * 1024 * 2.5)).toBe("2.5 MB");
  });

  it("formats duration", () => {
    expect(formatDurationSeconds(0)).toBe("—");
    expect(formatDurationSeconds(45)).toBe("45s");
    expect(formatDurationSeconds(90)).toBe("1m 30s");
    expect(formatDurationSeconds(3600)).toBe("1h 0m");
  });
});
