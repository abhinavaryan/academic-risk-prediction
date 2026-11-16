import type { Metadata } from "next";
import { PredictView } from "@/views/predict";

export const metadata: Metadata = { title: "Predict" };

export default function Page() {
  return <PredictView />;
}
