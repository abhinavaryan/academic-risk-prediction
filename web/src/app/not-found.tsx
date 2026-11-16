import Link from "next/link";
import { Container, buttonClass } from "@/components/ui";

export default function NotFound() {
  return (
    <Container className="py-24 text-center">
      <p className="text-sm font-medium text-accent-ink">404</p>
      <h1 className="mt-2 text-3xl font-semibold text-ink">This page does not exist</h1>
      <Link href="/" className={`${buttonClass.primary} mt-8`}>Back to the overview</Link>
    </Container>
  );
}
