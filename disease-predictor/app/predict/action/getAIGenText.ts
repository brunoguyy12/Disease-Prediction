"use server";

import { HfInference } from "@huggingface/inference";
import { GoogleGenerativeAI } from "@google/generative-ai";

export async function GetAIGenText(prediction: string) {
  console.log("is this running client or server side");
  try {
    const client = new HfInference(process.env.HF_Original_apiKey);

    const chatCompletion = await client.chatCompletion({
      model: "deepseek-ai/DeepSeek-R1",
      messages: [
        {
          role: "user",
          content:
            "Give a brief overview with just the description of the disease and a small cause (3sentences maximum) of the disease:  " +
            prediction,
        },
      ],
      provider: "fireworks-ai",
      // max_tokens: 500,
    });

    console.log(chatCompletion.choices[0].message);

    // Get the raw content
    let content = chatCompletion.choices[0].message.content;
    // Check for failure cases
    if (
      !content ||
      content === "I am sorry, I do not know the answer to that question."
    ) {
      return {
        success: false,
        message: "I am sorry, I do not know the answer to that question.",
      };
    }

    // Remove <think> section
    const thinkEndIndex = content.indexOf("</think>");
    if (thinkEndIndex !== -1) {
      content = content.substring(thinkEndIndex + 8).trim(); // +8 to skip '</think>'
    }

    // Format **{anything}** to bold using <strong> tags
    content = content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Trim and split into sentences, take up to 3
    const sentences = content
      .trim()
      .split(/(?<=\.)\s+/)
      .filter((s) => s.length > 0)
      .slice(0, 3)
      .join(" ");
    // Format **{anything}** to bold using <strong> tags

    // setMessage(chatCompletion.choices[0].message.content);
    return {
      success: true,
      message: sentences,
    };
  } catch (e) {
    console.error("Received eror: ", e);
    return {
      success: false,
      message: "I am sorry, I do not know the answer to that question.",
    };
  }
}

export async function GetGeminiGenText(prediction: string) {
  try {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const prompt = `Give a brief overview with just the description of the disease and a small cause (3 sentences maximum) of the disease: ${prediction}`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    // console.dir(response, { depth: null });
    const text = response.text();

    // console.log(text);
    if (!text) {
      return {
        success: false,
        message: "Failed to get a response from Gemini.",
      };
    }

    // Format **{anything}** to bold using <strong> tags
    const formattedText = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Trim and split into sentences, take up to 3
    const sentences = formattedText
      .trim()
      .split(/(?<=\.)\s+/)
      .filter((s) => s.length > 0)
      .slice(0, 3)
      .join(" ");

    return {
      success: true,
      message: sentences,
    };
  } catch (error) {
    console.error("Gemini API error:", error);
    return {
      success: false,
      message: "Failed to get information. Please try again later.",
    };
  }
}
