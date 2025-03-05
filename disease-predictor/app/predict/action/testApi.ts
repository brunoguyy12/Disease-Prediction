// "use server";

// import { GoogleGenerativeAI } from "@google/generative-ai";

// export async function listAvailableModels() {
//   try {
//     const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
//     const models = await genAI.getAvailableModels();
//     const modelAvailable = genAI.apiKey;
//     console.log("Available models:", models);
//     console.log("Key available:", modelAvailable);
//     return models;
//   } catch (error) {
//     console.error("Error listing models:", error);
//     return [];
//   }
// }

// // In your page or component:

// async function someFunction() {
//   const availableModels = await listAvailableModels();
//   console.log(availableModels);
// }

// someFunction();

// package.json:
// "testApi": "tsx app/predict/action/testApi.ts"
