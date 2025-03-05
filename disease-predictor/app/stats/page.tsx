// "use client";

// import { useState, useEffect } from "react";
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// import {
//   Card,
//   CardContent,
//   CardDescription,
//   CardHeader,
//   CardTitle,
// } from "@/components/ui/card";
// import { Skeleton } from "@/components/ui/skeleton";
// import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
// import {
//   AlertCircle,
//   BarChart3,
//   PieChart,
//   Network,
//   Loader2,
// } from "lucide-react";
// import {
//   Chart as ChartJS,
//   CategoryScale,
//   LinearScale,
//   BarElement,
//   Title,
//   Tooltip,
//   Legend,
//   ArcElement,
//   PointElement,
//   LineElement,
//   RadialLinearScale,
// } from "chart.js";
// import { Bar, Pie, Radar } from "react-chartjs-2";

// // Register ChartJS components
// ChartJS.register(
//   CategoryScale,
//   LinearScale,
//   BarElement,
//   ArcElement,
//   PointElement,
//   LineElement,
//   RadialLinearScale,
//   Title,
//   Tooltip,
//   Legend
// );

// // Types for our data
// interface DiseaseData {
//   [key: string]: number;
// }

// interface SymptomData {
//   [key: string]: number;
// }

// interface SymptomDiseaseRelation {
//   symptom: string;
//   diseases: string[];
//   count: number;
// }

// export default function StatsPage() {
//   const [diseaseData, setDiseaseData] = useState<DiseaseData>({});
//   const [symptomData, setSymptomData] = useState<SymptomData>({});
//   const [symptomDiseaseRelations, setSymptomDiseaseRelations] = useState<
//     SymptomDiseaseRelation[]
//   >([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   // Format text for display
//   const formatText = (text: string) => {
//     return text
//       .replace(/_/g, " ")
//       .replace(/\b\w/g, (char) => char.toUpperCase());
//   };

//   useEffect(() => {
//     const fetchData = async () => {
//       setLoading(true);
//       setError(null);

//       try {
//         // Fetch disease stats
//         const diseaseResponse = await fetch("http://localhost:5000/stats");
//         // await new Promise((resolve) => setTimeout(resolve, 10000));
//         if (!diseaseResponse.ok) {
//           throw new Error("Failed to fetch disease statistics");
//         }
//         const diseaseData = await diseaseResponse.json();
//         setDiseaseData(diseaseData);

//         // For demonstration, we'll generate symptom data and relations
//         // In a real app, you would fetch this from your backend
//         generateSymptomData();
//       } catch (err) {
//         console.error("Error fetching statistics:", err);
//         setError(
//           "Failed to load statistics. Please check if the server is running."
//         );
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, []);

//   // Generate sample symptom data for demonstration
//   // In a real app, this would come from your backend
//   const generateSymptomData = () => {
//     // Sample symptom frequency data
//     const sampleSymptomData: SymptomData = {
//       fever: 78,
//       cough: 65,
//       fatigue: 60,
//       headache: 55,
//       nausea: 48,
//       chest_pain: 42,
//       abdominal_pain: 40,
//       vomiting: 38,
//       dizziness: 35,
//       shortness_of_breath: 32,
//     };

//     setSymptomData(sampleSymptomData);

//     // Sample symptom-disease relations
//     const sampleRelations: SymptomDiseaseRelation[] = [
//       {
//         symptom: "fever",
//         diseases: ["Malaria", "Typhoid", "Influenza"],
//         count: 15,
//       },
//       {
//         symptom: "cough",
//         diseases: ["Common Cold", "Pneumonia", "Tuberculosis"],
//         count: 12,
//       },
//       {
//         symptom: "headache",
//         diseases: ["Migraine", "Hypertension", "Sinusitis"],
//         count: 10,
//       },
//       {
//         symptom: "chest_pain",
//         diseases: ["Heart attack", "Pneumonia", "GERD"],
//         count: 8,
//       },
//       {
//         symptom: "abdominal_pain",
//         diseases: ["Appendicitis", "Gastroenteritis", "Peptic ulcer"],
//         count: 7,
//       },
//     ];

//     setSymptomDiseaseRelations(sampleRelations);
//   };

//   // Prepare chart data
//   const diseaseChartData = {
//     labels: Object.keys(diseaseData).map(formatText),
//     datasets: [
//       {
//         label: "Disease Occurrences",
//         data: Object.values(diseaseData),
//         backgroundColor: "rgba(59, 130, 246, 0.6)",
//         borderColor: "rgba(59, 130, 246, 1)",
//         borderWidth: 1,
//       },
//     ],
//   };

//   const symptomChartData = {
//     labels: Object.keys(symptomData).map(formatText),
//     datasets: [
//       {
//         label: "Symptom Frequency",
//         data: Object.values(symptomData),
//         backgroundColor: [
//           "rgba(255, 99, 132, 0.6)",
//           "rgba(54, 162, 235, 0.6)",
//           "rgba(255, 206, 86, 0.6)",
//           "rgba(75, 192, 192, 0.6)",
//           "rgba(153, 102, 255, 0.6)",
//           "rgba(255, 159, 64, 0.6)",
//           "rgba(199, 199, 199, 0.6)",
//           "rgba(83, 102, 255, 0.6)",
//           "rgba(40, 159, 64, 0.6)",
//           "rgba(210, 199, 199, 0.6)",
//         ],
//         borderColor: [
//           "rgba(255, 99, 132, 1)",
//           "rgba(54, 162, 235, 1)",
//           "rgba(255, 206, 86, 1)",
//           "rgba(75, 192, 192, 1)",
//           "rgba(153, 102, 255, 1)",
//           "rgba(255, 159, 64, 1)",
//           "rgba(199, 199, 199, 1)",
//           "rgba(83, 102, 255, 1)",
//           "rgba(40, 159, 64, 1)",
//           "rgba(210, 199, 199, 1)",
//         ],
//         borderWidth: 1,
//       },
//     ],
//   };

//   // Prepare radar chart data for disease complexity
//   const complexityData = {
//     labels: [
//       "Malaria",
//       "Tuberculosis",
//       "Diabetes",
//       "Hypertension",
//       "Asthma",
//       "Arthritis",
//     ],
//     datasets: [
//       {
//         label: "Average Number of Symptoms",
//         data: [8, 6, 5, 4, 7, 3],
//         backgroundColor: "rgba(59, 130, 246, 0.2)",
//         borderColor: "rgba(59, 130, 246, 1)",
//         borderWidth: 1,
//         pointBackgroundColor: "rgba(59, 130, 246, 1)",
//         pointBorderColor: "#fff",
//         pointHoverBackgroundColor: "#fff",
//         pointHoverBorderColor: "rgba(59, 130, 246, 1)",
//       },
//     ],
//   };

//   const chartOptions = {
//     responsive: true,
//     plugins: {
//       legend: { position: "top" as const },
//       title: { display: true, text: "Disease Statistics" },
//     },
//   };

//   const pieChartOptions = {
//     responsive: true,
//     plugins: {
//       legend: {
//         position: "right" as const,
//         labels: {
//           boxWidth: 15,
//         },
//       },
//       title: { display: true, text: "Symptom Frequency" },
//     },
//   };

//   const radarOptions = {
//     responsive: true,
//     plugins: {
//       legend: { position: "top" as const },
//       title: { display: true, text: "Disease Complexity (Number of Symptoms)" },
//     },
//     scales: {
//       r: {
//         min: 0,
//         max: 10,
//         ticks: {
//           stepSize: 2,
//         },
//       },
//     },
//   };

//   return (
//     <div className="container mx-auto py-10 px-4">
//       <div className="max-w-5xl mx-auto space-y-8">
//         <div className="text-center space-y-2">
//           <h1 className="text-3xl font-bold">Disease Statistics</h1>
//           <p className="text-muted-foreground">
//             Explore insights about diseases, symptoms, and their relationships
//           </p>
//         </div>

//         {error && (
//           <Alert variant="destructive">
//             <AlertCircle className="h-4 w-4" />
//             <AlertTitle>Error</AlertTitle>
//             <AlertDescription>{error}</AlertDescription>
//           </Alert>
//         )}

//         {loading ? (
//           <div className="space-y-4">
//             <div className="flex items-center justify-center space-x-2">
//               <Loader2 className="h-6 w-6 animate-spin text-primary" />
//               <p>Loading statistics...</p>
//             </div>
//             <Skeleton className="h-[400px] w-full" />
//           </div>
//         ) : (
//           <Tabs defaultValue="overview" className="w-full">
//             <TabsList className="grid w-full grid-cols-3">
//               <TabsTrigger value="overview" className="flex items-center gap-2">
//                 <BarChart3 className="h-4 w-4" />
//                 <span>Disease Overview</span>
//               </TabsTrigger>
//               <TabsTrigger value="symptoms" className="flex items-center gap-2">
//                 <PieChart className="h-4 w-4" />
//                 <span>Symptom Analysis</span>
//               </TabsTrigger>
//               <TabsTrigger
//                 value="relations"
//                 className="flex items-center gap-2"
//               >
//                 <Network className="h-4 w-4" />
//                 <span>Relationships</span>
//               </TabsTrigger>
//             </TabsList>

//             <TabsContent value="overview" className="mt-6">
//               <Card>
//                 <CardHeader>
//                   <CardTitle>Disease Distribution</CardTitle>
//                   <CardDescription>
//                     Overview of diseases in our dataset
//                   </CardDescription>
//                 </CardHeader>
//                 <CardContent>
//                   <div className="h-[400px]">
//                     <Bar data={diseaseChartData} options={chartOptions} />
//                   </div>
//                 </CardContent>
//               </Card>
//             </TabsContent>

//             <TabsContent value="symptoms" className="mt-6 space-y-6">
//               <Card>
//                 <CardHeader>
//                   <CardTitle>Symptom Frequency</CardTitle>
//                   <CardDescription>
//                     Most common symptoms across all diseases
//                   </CardDescription>
//                 </CardHeader>
//                 <CardContent>
//                   <div className="h-[400px]">
//                     <Pie data={symptomChartData} options={pieChartOptions} />
//                   </div>
//                 </CardContent>
//               </Card>

//               <Card>
//                 <CardHeader>
//                   <CardTitle>Disease Complexity</CardTitle>
//                   <CardDescription>
//                     Number of symptoms typically associated with each disease
//                   </CardDescription>
//                 </CardHeader>
//                 <CardContent>
//                   <div className="h-[400px]">
//                     <Radar data={complexityData} options={radarOptions} />
//                   </div>
//                 </CardContent>
//               </Card>
//             </TabsContent>

//             <TabsContent value="relations" className="mt-6">
//               <Card>
//                 <CardHeader>
//                   <CardTitle>Symptom-Disease Relationships</CardTitle>
//                   <CardDescription>
//                     Which symptoms are most predictive of which diseases
//                   </CardDescription>
//                 </CardHeader>
//                 <CardContent>
//                   <div className="space-y-6">
//                     {symptomDiseaseRelations.map((relation) => (
//                       <div
//                         key={relation.symptom}
//                         className="border rounded-lg p-4"
//                       >
//                         <h3 className="text-lg font-medium mb-2">
//                           {formatText(relation.symptom)}
//                         </h3>
//                         <div className="flex flex-wrap gap-2 mb-2">
//                           {relation.diseases.map((disease) => (
//                             <span
//                               key={disease}
//                               className="px-2 py-1 bg-primary/10 text-primary rounded-full text-sm"
//                             >
//                               {disease}
//                             </span>
//                           ))}
//                         </div>
//                         <p className="text-sm text-muted-foreground">
//                           Associated with {relation.count} disease patterns
//                         </p>
//                       </div>
//                     ))}
//                   </div>
//                 </CardContent>
//               </Card>
//             </TabsContent>
//           </Tabs>
//         )}

//         <Card className="bg-muted/40">
//           <CardHeader>
//             <CardTitle>About the Statistics</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <p className="text-sm text-muted-foreground">
//               These statistics are derived from our disease prediction
//               model&apos;s training data. The visualizations show disease
//               distribution, symptom frequency, disease complexity (number of
//               symptoms per disease), and symptom-disease relationships. This
//               information can help understand patterns in disease diagnosis and
//               symptom presentation.
//             </p>
//           </CardContent>
//         </Card>
//       </div>
//     </div>
//   );
// }

"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertCircle,
  BarChart3,
  PieChart,
  Network,
  Loader2,
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  RadialLinearScale,
} from "chart.js";
import { Bar, Pie, Radar } from "react-chartjs-2";
import { useMediaQuery } from "@/lib/use-media-query";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
);

// Types for our data
interface DiseaseData {
  [key: string]: number;
}

interface SymptomData {
  [key: string]: number;
}

interface SymptomDiseaseRelation {
  symptom: string;
  diseases: string[];
  count: number;
}

export default function StatsPage() {
  const [diseaseData, setDiseaseData] = useState<DiseaseData>({});
  const [symptomData, setSymptomData] = useState<SymptomData>({});
  const [symptomDiseaseRelations, setSymptomDiseaseRelations] = useState<
    SymptomDiseaseRelation[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const isMobile = useMediaQuery("(max-width: 768px)");

  // Format text for display
  const formatText = (text: string) => {
    return text
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch disease stats
        const diseaseResponse = await fetch("http://localhost:5000/stats");
        if (!diseaseResponse.ok) {
          throw new Error("Failed to fetch disease statistics");
        }
        const diseaseData = await diseaseResponse.json();
        setDiseaseData(diseaseData);

        // For demonstration, we'll generate symptom data and relations
        // In a real app, you would fetch this from your backend
        generateSymptomData();
      } catch (err) {
        console.error("Error fetching statistics:", err);
        setError(
          "Failed to load statistics. Please check if the server is running."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Generate sample symptom data for demonstration
  // In a real app, this would come from your backend
  const generateSymptomData = () => {
    // Sample symptom frequency data
    const sampleSymptomData: SymptomData = {
      fever: 78,
      cough: 65,
      fatigue: 60,
      headache: 55,
      nausea: 48,
      chest_pain: 42,
      abdominal_pain: 40,
      vomiting: 38,
      dizziness: 35,
      shortness_of_breath: 32,
    };

    setSymptomData(sampleSymptomData);

    // Sample symptom-disease relations
    const sampleRelations: SymptomDiseaseRelation[] = [
      {
        symptom: "fever",
        diseases: ["Malaria", "Typhoid", "Influenza"],
        count: 15,
      },
      {
        symptom: "cough",
        diseases: ["Common Cold", "Pneumonia", "Tuberculosis"],
        count: 12,
      },
      {
        symptom: "headache",
        diseases: ["Migraine", "Hypertension", "Sinusitis"],
        count: 10,
      },
      {
        symptom: "chest_pain",
        diseases: ["Heart attack", "Pneumonia", "GERD"],
        count: 8,
      },
      {
        symptom: "abdominal_pain",
        diseases: ["Appendicitis", "Gastroenteritis", "Peptic ulcer"],
        count: 7,
      },
    ];

    setSymptomDiseaseRelations(sampleRelations);
  };

  // Prepare chart data
  const diseaseChartData = {
    labels: Object.keys(diseaseData).map(formatText),
    datasets: [
      {
        label: "Disease Occurrences",
        data: Object.values(diseaseData),
        backgroundColor: "rgba(59, 130, 246, 0.6)",
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 1,
      },
    ],
  };

  const symptomChartData = {
    labels: Object.keys(symptomData).map(formatText),
    datasets: [
      {
        label: "Symptom Frequency",
        data: Object.values(symptomData),
        backgroundColor: [
          "rgba(255, 99, 132, 0.6)",
          "rgba(54, 162, 235, 0.6)",
          "rgba(255, 206, 86, 0.6)",
          "rgba(75, 192, 192, 0.6)",
          "rgba(153, 102, 255, 0.6)",
          "rgba(255, 159, 64, 0.6)",
          "rgba(199, 199, 199, 0.6)",
          "rgba(83, 102, 255, 0.6)",
          "rgba(40, 159, 64, 0.6)",
          "rgba(210, 199, 199, 0.6)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(153, 102, 255, 1)",
          "rgba(255, 159, 64, 1)",
          "rgba(199, 199, 199, 1)",
          "rgba(83, 102, 255, 1)",
          "rgba(40, 159, 64, 1)",
          "rgba(210, 199, 199, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };

  // Prepare radar chart data for disease complexity
  const complexityData = {
    labels: [
      "Malaria",
      "Tuberculosis",
      "Diabetes",
      "Hypertension",
      "Asthma",
      "Arthritis",
    ],
    datasets: [
      {
        label: "Average Number of Symptoms",
        data: [8, 6, 5, 4, 7, 3],
        backgroundColor: "rgba(59, 130, 246, 0.2)",
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 1,
        pointBackgroundColor: "rgba(59, 130, 246, 1)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(59, 130, 246, 1)",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: isMobile ? ("bottom" as const) : ("top" as const),
        labels: {
          boxWidth: isMobile ? 10 : 15,
          font: {
            size: isMobile ? 10 : 12,
          },
        },
      },
      title: {
        display: true,
        text: "Disease Statistics",
        font: {
          size: isMobile ? 14 : 16,
        },
      },
    },
    scales: {
      x: {
        ticks: {
          maxRotation: isMobile ? 90 : 0,
          minRotation: isMobile ? 90 : 0,
          font: {
            size: isMobile ? 8 : 12,
          },
        },
      },
      y: {
        ticks: {
          font: {
            size: isMobile ? 8 : 12,
          },
        },
      },
    },
  };

  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: isMobile ? ("bottom" as const) : ("right" as const),
        labels: {
          boxWidth: isMobile ? 10 : 15,
          font: {
            size: isMobile ? 10 : 12,
          },
        },
      },
      title: {
        display: true,
        text: "Symptom Frequency",
        font: {
          size: isMobile ? 14 : 16,
        },
      },
    },
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: isMobile ? ("bottom" as const) : ("top" as const),
        labels: {
          boxWidth: isMobile ? 10 : 15,
          font: {
            size: isMobile ? 10 : 12,
          },
        },
      },
      title: {
        display: true,
        text: "Disease Complexity",
        font: {
          size: isMobile ? 14 : 16,
        },
      },
    },
    scales: {
      r: {
        min: 0,
        max: 10,
        ticks: {
          stepSize: 2,
          font: {
            size: isMobile ? 8 : 10,
          },
        },
        pointLabels: {
          font: {
            size: isMobile ? 8 : 10,
          },
        },
      },
    },
  };

  return (
    <div className="container mx-auto py-6 px-4">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-2xl md:text-3xl font-bold">Disease Statistics</h1>
          <p className="text-sm md:text-base text-muted-foreground">
            Explore insights about diseases, symptoms, and their relationships
          </p>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center space-x-2">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p>Loading statistics...</p>
            </div>
            <Skeleton className="h-[300px] md:h-[400px] w-full" />
          </div>
        ) : (
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger
                value="overview"
                className="flex items-center gap-1 md:gap-2 text-xs md:text-sm"
              >
                <BarChart3 className="h-3 w-3 md:h-4 md:w-4" />
                <span className="hidden xs:inline">Disease</span> Overview
              </TabsTrigger>
              <TabsTrigger
                value="symptoms"
                className="flex items-center gap-1 md:gap-2 text-xs md:text-sm"
              >
                <PieChart className="h-3 w-3 md:h-4 md:w-4" />
                <span className="hidden xs:inline">Symptom</span> Analysis
              </TabsTrigger>
              <TabsTrigger
                value="relations"
                className="flex items-center gap-1 md:gap-2 text-xs md:text-sm"
              >
                <Network className="h-3 w-3 md:h-4 md:w-4" />
                Relationships
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-4 md:mt-6">
              <Card>
                <CardHeader className="pb-2 md:pb-4">
                  <CardTitle className="text-lg md:text-xl">
                    Disease Distribution
                  </CardTitle>
                  <CardDescription className="text-xs md:text-sm">
                    Overview of diseases in our dataset
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] md:h-[400px] w-full">
                    <Bar data={diseaseChartData} options={chartOptions} />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent
              value="symptoms"
              className="mt-4 md:mt-6 space-y-4 md:space-y-6"
            >
              <Card>
                <CardHeader className="pb-2 md:pb-4">
                  <CardTitle className="text-lg md:text-xl">
                    Symptom Frequency
                  </CardTitle>
                  <CardDescription className="text-xs md:text-sm">
                    Most common symptoms across all diseases
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] md:h-[400px] w-full">
                    <Pie data={symptomChartData} options={pieChartOptions} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2 md:pb-4">
                  <CardTitle className="text-lg md:text-xl">
                    Disease Complexity
                  </CardTitle>
                  <CardDescription className="text-xs md:text-sm">
                    Number of symptoms typically associated with each disease
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] md:h-[400px] w-full">
                    <Radar data={complexityData} options={radarOptions} />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="relations" className="mt-4 md:mt-6">
              <Card>
                <CardHeader className="pb-2 md:pb-4">
                  <CardTitle className="text-lg md:text-xl">
                    Symptom-Disease Relationships
                  </CardTitle>
                  <CardDescription className="text-xs md:text-sm">
                    Which symptoms are most predictive of which diseases
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {symptomDiseaseRelations.map((relation) => (
                      <div
                        key={relation.symptom}
                        className="border rounded-lg p-3 md:p-4"
                      >
                        <h3 className="text-base md:text-lg font-medium mb-2">
                          {formatText(relation.symptom)}
                        </h3>
                        <div className="flex flex-wrap gap-1 md:gap-2 mb-2">
                          {relation.diseases.map((disease) => (
                            <span
                              key={disease}
                              className="px-2 py-1 bg-primary/10 text-primary rounded-full text-xs md:text-sm"
                            >
                              {disease}
                            </span>
                          ))}
                        </div>
                        <p className="text-xs md:text-sm text-muted-foreground">
                          Associated with {relation.count} disease patterns
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}

        <Card className="bg-muted/40">
          <CardHeader className="pb-2 md:pb-4">
            <CardTitle className="text-lg md:text-xl">
              About the Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs md:text-sm text-muted-foreground">
              These statistics are derived from our disease prediction
              model&apos;s training data. The visualizations show disease
              distribution, symptom frequency, disease complexity (number of
              symptoms per disease), and symptom-disease relationships. This
              information can help understand patterns in disease diagnosis and
              symptom presentation.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
