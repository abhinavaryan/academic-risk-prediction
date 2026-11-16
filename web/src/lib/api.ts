import type { CustomResult, DatasetProfile, Meta, ModelSchema, Prediction, Sample, TrainConfig } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public fields?: Record<string, string>,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network", "Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let body: { error?: { code?: string; message?: string; fields?: Record<string, string> } } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error page */
    }
    const retry = res.headers.get("retry-after");
    const message =
      body.error?.message ??
      (res.status === 413 ? "The file is too large." : `Request failed (${res.status}).`) +
        (retry ? ` Try again in ${retry}s.` : "");
    throw new ApiError(res.status, body.error?.code ?? "http_error", message, body.error?.fields);
  }
  return (await res.json()) as T;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  schema: (cutoff: number) => request<ModelSchema>(`/api/schema/${cutoff}`),
  samples: (cutoff: number) => request<Sample[]>(`/api/samples/${cutoff}`),
  predict: (cutoff: number, features: Record<string, unknown>, signal?: AbortSignal) =>
    request<Prediction>("/api/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cutoff, features }),
      signal,
    }),
  profile: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<DatasetProfile>("/api/custom/profile", { method: "POST", body });
  },
  train: (file: File, config: TrainConfig) => {
    const body = new FormData();
    body.append("file", file);
    body.append("config", JSON.stringify(config));
    return request<CustomResult>("/api/custom/train", { method: "POST", body });
  },
  sampleFile: async (): Promise<File> => {
    const res = await fetch(`${BASE}/api/custom/sample`);
    if (!res.ok) throw new ApiError(res.status, "http_error", "Could not load the sample dataset.");
    return new File([await res.blob()], "uci_student_dropout.csv", { type: "text/csv" });
  },
};
