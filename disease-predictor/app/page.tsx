import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Activity, BarChart3, HeartPulse, Stethoscope } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="container mx-auto py-10 px-4 space-y-12">
      <section className="flex flex-col items-center text-center space-y-4 py-10">
        <div className="inline-block p-4 rounded-full bg-primary/10 mb-4">
          <HeartPulse className="h-12 w-12 text-primary" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          Disease Prediction System
        </h1>
        <p className="text-xl text-muted-foreground max-w-[700px]">
          Identify potential health conditions based on your symptoms using
          advanced machine learning.
        </p>
        <div className="flex flex-wrap gap-4 mt-6 justify-center">
          <Button asChild size="lg" className="gap-2">
            <Link href="/predict">
              <Activity className="h-5 w-5" />
              Start Prediction
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="gap-2">
            <Link href="/stats">
              <BarChart3 className="h-5 w-5" />
              View Statistics
            </Link>
          </Button>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="overflow-hidden">
          <CardHeader className="bg-primary/5 pb-4">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Disease Prediction
            </CardTitle>
            <CardDescription>
              Select your symptoms to get a prediction
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <p>
                Our advanced machine learning algorithm analyzes your symptoms
                to identify potential health conditions with high accuracy.
              </p>
              <ul className="list-disc pl-5 space-y-1">
                <li>Select from over 100 different symptoms</li>
                <li>Get instant predictions</li>
                <li>Understand potential health risks</li>
              </ul>
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild className="w-full">
              <Link href="/predict">Go to Prediction</Link>
            </Button>
          </CardFooter>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader className="bg-primary/5 pb-4">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Health Statistics
            </CardTitle>
            <CardDescription>Explore health data and insights</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <p>
                Gain valuable insights into disease patterns, symptom
                relationships, and health trends through interactive
                visualizations.
              </p>
              <ul className="list-disc pl-5 space-y-1">
                <li>Symptom frequency analysis</li>
                <li>Disease complexity metrics</li>
                <li>Symptom correlation insights</li>
              </ul>
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild variant="outline" className="w-full">
              <Link href="/stats">View Statistics</Link>
            </Button>
          </CardFooter>
        </Card>
      </section>

      <section className="text-center py-8">
        <div className="inline-block p-3 rounded-full bg-primary/10 mb-4">
          <Stethoscope className="h-8 w-8 text-primary" />
        </div>
        <h2 className="text-2xl font-bold mb-4">Important Disclaimer</h2>
        <p className="text-muted-foreground max-w-[700px] mx-auto">
          This tool is for informational purposes only and should not replace
          professional medical advice. Always consult with a healthcare provider
          for proper diagnosis and treatment.
        </p>
      </section>
    </div>
  );
}
