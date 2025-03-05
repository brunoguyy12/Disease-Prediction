"use client";

// import { OpenAI } from "openai";
import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Check,
  ChevronsUpDown,
  AlertCircle,
  Loader2,
  X,
  MoveUpRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import Link from "next/link";
import {
  // GetAIGenText,
  GetGeminiGenText,
} from "@/app/predict/action/getAIGenText";

export default function PredictPage() {
  const [symptomsList, setSymptomsList] = useState<string[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchingSymptoms, setFetchingSymptoms] = useState(true);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [message, setMessage] = useState<string>("");
  const [genLoading, setGenLoading] = useState<boolean>(false);
  const [genAiError, setGenAiError] = useState<string | null>(null);

  // Format symptom for display
  const formatSymptom = (symptom: string) => {
    return symptom
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  // Fetch available symptoms on mount
  useEffect(() => {
    setFetchingSymptoms(true);

    const fetchSymptoms = async () => {
      try {
        const res = await fetch("http://localhost:5000/symptoms");
        // await new Promise((resolve) => setTimeout(resolve, 10000));
        if (!res.ok) throw new Error("Failed to fetch symptoms");
        const data = await res.json();
        setSymptomsList(data.symptoms);
      } catch (err) {
        console.error("Error fetching symptoms:", err);
        setError(
          "Failed to load symptoms. Please check if the server is running."
        );
      } finally {
        setFetchingSymptoms(false);
      }
    };

    fetchSymptoms();
  }, []);

  // Handle symptom selection
  const handleSymptomSelect = (value: string) => {
    if (!selectedSymptoms.includes(value)) {
      setSelectedSymptoms((prev) => [...prev, value]);
    }
    setOpen(false);
  };

  // Remove symptom
  const removeSymptom = (symptom: string) => {
    setSelectedSymptoms(selectedSymptoms.filter((s) => s !== symptom));
  };

  // Submit symptoms for prediction
  const handlePredict = async () => {
    if (selectedSymptoms.length === 0) {
      setError("Please select at least one symptom");
      toast.error("Please select at least one symptom");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: selectedSymptoms }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();
      setPrediction(data.disease);
    } catch (error) {
      console.error("Prediction error:", error);
      setError("Failed to get prediction. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleInformation = async () => {
    try {
      setGenLoading(true);
      setGenAiError(null); // clear previous errors
      // const res = await GetAIGenText(prediction!);

      const res = await GetGeminiGenText(prediction!);

      if (res.success) {
        setMessage(res.message);
      } else {
        setGenAiError("Failed to get information. Please try again later.");
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      setGenAiError(
        "Failed to get information. Please try again later." + e?.message
      );
    } finally {
      setGenLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-10 px-4">
      <div className="max-w-3xl mx-auto space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Disease Prediction</h1>
          <p className="text-muted-foreground">
            Select your symptoms to get a potential diagnosis
          </p>
        </div>

        {error && (
          <div className="relative">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
            <X
              className="h-4 w-4 absolute top-2 text-destructive right-2 cursor-pointer"
              onClick={() => setError(null)}
            />
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Select Symptoms</CardTitle>
            <CardDescription>
              Choose all symptoms you are experiencing
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={open}
                  className="w-full justify-between"
                  disabled={fetchingSymptoms}
                >
                  {fetchingSymptoms ? (
                    <span className="flex items-center">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading symptoms...
                    </span>
                  ) : (
                    <>
                      <span>Select a symptom</span>
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-full p-0">
                <Command>
                  <CommandInput placeholder="Search symptoms..." />
                  <CommandList>
                    <CommandEmpty>No symptom found.</CommandEmpty>
                    <CommandGroup className="max-h-[300px] overflow-y-auto">
                      {symptomsList.map((symptom) => (
                        <CommandItem
                          key={symptom}
                          value={symptom}
                          onSelect={() => handleSymptomSelect(symptom)}
                        >
                          <Check
                            className={cn(
                              "mr-2 h-4 w-4",
                              selectedSymptoms.includes(symptom)
                                ? "opacity-100"
                                : "opacity-0"
                            )}
                          />
                          {formatSymptom(symptom)}
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>

            <div>
              <h3 className="text-sm font-medium mb-2">Selected Symptoms:</h3>
              {selectedSymptoms.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {selectedSymptoms.map((symptom) => (
                    <Badge
                      key={symptom}
                      variant="secondary"
                      className="flex items-center gap-1"
                    >
                      {formatSymptom(symptom)}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-4 w-4 p-0 ml-1 cursor-pointer"
                        onClick={() => removeSymptom(symptom)}
                      >
                        <X className="h-3 w-3" />
                        <span className="sr-only">Remove</span>
                      </Button>
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No symptoms selected.
                </p>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={handlePredict}
              disabled={loading || fetchingSymptoms}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing Symptoms...
                </>
              ) : (
                "Predict Disease"
              )}
            </Button>
          </CardFooter>
        </Card>

        {prediction && (
          <Card className="border-primary/50">
            <CardHeader className="bg-primary/5 ">
              <CardTitle>Prediction Result</CardTitle>
              <CardDescription>
                Based on your symptoms, our model predicts:
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="text-center py-4">
                <Link
                  href={`https://www.google.com/search?q=${prediction}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-2xl font-bold text-primary mb-2 flex justify-center gap-2 group"
                >
                  {prediction}
                  <MoveUpRight className="h-6 w-6 text-primary group-hover:translate-x-1 group-hover:-translate-y-1 transition-all" />
                </Link>
                <p className="text-sm text-muted-foreground">
                  This prediction is based on the symptoms you selected and our
                  machine learning model.
                </p>
                <Button
                  onClick={() => {
                    handleInformation();
                  }}
                  disabled={genLoading}
                  className="mt-4 w-full"
                >
                  {genLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading Information...
                    </>
                  ) : (
                    "Describe the Disease"
                  )}
                </Button>

                {genLoading && (
                  <div className="mt-4">
                    <Skeleton className="h-[200px] w-full" />
                  </div>
                )}
                {genAiError && (
                  <div className="relative mt-4">
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Error</AlertTitle>
                      <AlertDescription>{genAiError}</AlertDescription>
                    </Alert>
                    <X
                      className="h-4 w-4 absolute top-2 text-destructive right-2 cursor-pointer"
                      onClick={() => setError(null)}
                    />
                  </div>
                )}
                {!genLoading && message && (
                  <div className="mt-4">
                    <Alert variant="info">
                      <AlertTitle className="text-[18px] font-semibold mb-2">
                        Information about the Disease
                      </AlertTitle>
                      <AlertDescription
                        dangerouslySetInnerHTML={{ __html: message }}
                      />
                    </Alert>
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex flex-col space-y-2 items-start">
              <div className="text-sm text-muted-foreground">
                <strong>Disclaimer:</strong> This is not a medical diagnosis.
                Please consult with a healthcare professional.
              </div>
              <Button
                variant="outline"
                onClick={() => {
                  setPrediction(null);
                  setSelectedSymptoms([]);
                  setMessage("");
                  setError(null);
                  setGenAiError(null);
                }}
                className="w-full"
              >
                Start New Prediction
              </Button>
            </CardFooter>
          </Card>
        )}

        {fetchingSymptoms && (
          <div className="space-y-3">
            <Skeleton className="h-[40px] w-full" />
            <Skeleton className="h-[100px] w-full" />
            <Skeleton className="h-[40px] w-full" />
          </div>
        )}
      </div>
    </div>
  );
}
