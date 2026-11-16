import type { Metadata } from "next";
import { TrainView } from "@/views/train";

export const metadata: Metadata = { title: "Train on your data" };

export default function Page() {
  return <TrainView />;
}
