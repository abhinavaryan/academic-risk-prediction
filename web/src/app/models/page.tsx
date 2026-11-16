import type { Metadata } from "next";
import { ModelsView } from "@/views/models";

export const metadata: Metadata = { title: "Models" };

export default function Page() {
  return <ModelsView />;
}
