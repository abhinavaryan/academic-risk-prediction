import type { Metadata } from "next";
import { DatasetView } from "@/views/dataset";

export const metadata: Metadata = { title: "Dataset" };

export default function Page() {
  return <DatasetView />;
}
