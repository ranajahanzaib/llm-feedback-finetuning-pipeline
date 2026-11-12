// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main";

const dataset = {
  id: "ds-1",
  manifest: {
    counts: { train: 8, valid: 1, test: 1 },
    warnings: [],
    file_sha256: {},
    schema_version: "1",
    statistics: {},
  },
  created_at: "2026-01-01T00:00:00Z",
};
const candidate = {
  id: "model-1",
  base_model: "local/model",
  status: "evaluated",
  metrics: {},
  dataset_id: "ds-1",
  evaluation_id: "eval-1",
};
const report = {
  baseline: {
    exact_match_accuracy: 0.5,
    per_field_accuracy: {},
    n: 2,
    examples: [{ prediction: {} }, { prediction: {} }],
  },
  candidate: {
    exact_match_accuracy: 1,
    schema_adherence: 1,
    per_field_accuracy: {},
    n: 2,
    examples: [
      { input_text: "Item A", ground_truth: {}, prediction: {} },
      { input_text: "Item B", ground_truth: {}, prediction: {} },
    ],
  },
  prompting_only: { exact_match_accuracy: 0.5, latency_ms_median: 10 },
  improved: 1,
  regressed: 0,
  both_correct: 1,
  both_wrong: 0,
  cases: [{ outcome: "improved" }, { outcome: "both_correct" }],
  paired_statistics: { confidence_interval: [0, 1], seed: 42 },
};

function response(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

describe("operator workflows", () => {
  let feedback: any[];
  let datasets: any[];
  let models: any[];
  let evaluations: any[];
  let experiments: any[];
  let deployments: any[];
  let active: any;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    feedback = [
      {
        id: "fb-1",
        input_text: "Cotton shirt",
        original_prediction: {},
        corrected_output: { category: "shirt" },
        status: "pending",
        source: "import",
        privacy_flags: [],
        conflict: 0,
      },
    ];
    datasets = [dataset];
    models = [candidate];
    evaluations = [];
    experiments = [];
    deployments = [];
    active = null;
    fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      if (method === "GET") {
        if (url.endsWith("/feedback")) return response(feedback);
        if (url.endsWith("/datasets")) return response(datasets);
        if (url.endsWith("/evaluation-sets"))
          return response([
            {
              id: "holdout-1",
              manifest: { count: 2, created_at: "2026-01-01T00:00:00Z" },
            },
          ]);
        if (url.endsWith("/models/active")) return response(active);
        if (url.endsWith("/models")) return response(models);
        if (url.endsWith("/evaluations")) return response(evaluations);
        if (url.endsWith("/experiments")) return response(experiments);
        if (url.endsWith("/deployments")) return response(deployments);
      }
      const body = init?.body ? JSON.parse(String(init.body)) : {};
      if (url.endsWith("/feedback/fb-1/review")) {
        feedback = feedback.map((row) => ({ ...row, status: body.decision }));
        return response({});
      }
      if (url.endsWith("/datasets/build")) {
        datasets = [{ ...dataset, id: "ds-new" }, ...datasets];
        return response(datasets[0]);
      }
      if (url.endsWith("/train/jobs")) {
        experiments = [
          {
            id: "job-1",
            status: "queued",
            dataset_id: body.dataset_id,
            config: {},
            manifest: {},
          },
          ...experiments,
        ];
        return response(experiments[0]);
      }
      if (url.endsWith("/evaluations")) {
        evaluations = [{ id: "eval-2", report }, ...evaluations];
        return response(evaluations[0]);
      }
      if (url.endsWith("/models/model-1/promote")) {
        active = { ...candidate, status: "active" };
        deployments = [
          {
            id: "dep-1",
            action: "promoted",
            model_id: "model-1",
            created_at: "2026-01-01T00:00:00Z",
            details: {},
          },
          ...deployments,
        ];
        return response(active);
      }
      if (url.endsWith("/models/rollback")) {
        active = null;
        deployments = [
          {
            id: "dep-2",
            action: "rollback",
            model_id: "model-1",
            created_at: "2026-01-01T00:00:00Z",
            details: {},
          },
          ...deployments,
        ];
        return response({});
      }
      return response({ detail: `Unexpected request: ${method} ${url}` }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  async function openPage(name: string) {
    render(<App />);
    await screen.findByText("LOCAL MODE");
    fireEvent.click(screen.getByRole("button", { name: new RegExp(name) }));
    await screen.findByRole("heading", { name });
  }

  it("approves a reviewed correction and refreshes its status", async () => {
    await openPage("Corrections");
    const row = screen.getByText("Cotton shirt").closest("tr")!;
    fireEvent.click(within(row).getByRole("button", { name: /Approve/ }));
    await waitFor(() => expect(within(row).getByText("approved")).toBeTruthy());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/feedback/fb-1/review",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("creates a dataset version from the dataset page", async () => {
    await openPage("Datasets");
    fireEvent.click(screen.getByRole("button", { name: /Create dataset/ }));
    expect(await screen.findByText("ds-new")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/datasets/build",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("submits an experiment for the selected dataset", async () => {
    await openPage("Experiments");
    await waitFor(() =>
      expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe(
        "ds-1",
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /Start smoke LoRA job/ }),
    );
    expect(await screen.findByText("job-1")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/train/jobs",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("runs and displays a paired evaluation on the chosen holdout", async () => {
    await openPage("Evaluation");
    const selects = screen.getAllByRole("combobox") as HTMLSelectElement[];
    fireEvent.change(selects[0], { target: { value: "model-1" } });
    fireEvent.change(selects[1], { target: { value: "holdout-1" } });
    fireEvent.click(
      screen.getByRole("button", { name: /Run real evaluation/ }),
    );
    expect(await screen.findByText("HELD-OUT EXAMPLE INSPECTION")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/evaluations",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("holdout-1"),
      }),
    );
  });

  it("confirms promotion and rollback through registry controls", async () => {
    await openPage("Registry");
    fireEvent.click(screen.getByRole("button", { name: /Promote/ }));
    expect(await screen.findByText("Active deployment")).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: /Rollback deployment/ }),
    );
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/models/rollback",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    expect(window.confirm).toHaveBeenCalledTimes(2);
  });
});
