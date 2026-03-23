import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "../App";

describe("App", () => {
  it("renders core sections", () => {
    render(<App />);
    expect(screen.getByText("VideoWala PoC Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Profiles")).toBeInTheDocument();
    expect(screen.getByText("Events")).toBeInTheDocument();
    expect(screen.getByText("Event Summary")).toBeInTheDocument();
    expect(screen.getByText("Ingest media")).toBeInTheDocument();
    expect(screen.getByText("Photo curation")).toBeInTheDocument();
    expect(screen.getByText("People & face references")).toBeInTheDocument();
    expect(screen.getByText("Video plan + render")).toBeInTheDocument();
    expect(screen.getByText("Render Jobs")).toBeInTheDocument();
  });
});
