import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "../App";

describe("App", () => {
  it("renders core sections", () => {
    render(<App />);
    expect(screen.getByText("VideoWala MVP Frontend")).toBeInTheDocument();
    expect(screen.getByText("Event")).toBeInTheDocument();
    expect(screen.getByText("Ingest + Context")).toBeInTheDocument();
    expect(screen.getByText("Plan + Render + Regenerate")).toBeInTheDocument();
    expect(screen.getByText("Face APIs")).toBeInTheDocument();
  });
});
