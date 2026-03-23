"use client";

import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const predictSentiment = async () => {
    if (!text) return;

    setLoading(true);
    setError(""); // clear previous error
    setSentiment(""); // clear previous result

    try {
      const res = await fetch("https://anshidam-nlp-project.hf.space/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text }),
      });

      if (!res.ok) {
        throw new Error("API error");
      }

      const data = await res.json();
      setSentiment(data.sentiment);
    } catch (err) {
      console.error(err);
      setError("Error connecting to API");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-cover flex items-center justify-center">
      {/* Glass / overlay card */}
      <div className="bg-black/60 backdrop-blur-md p-8 rounded-2xl w-full max-w-xl text-white text-center shadow-lg">
        <h1 className="text-3xl font-bold mb-6">Sentiment Analyzer</h1>

        <textarea
          placeholder="Enter text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
          className="w-full p-3 bg-white text-black rounded-lg outline-none focus:ring-2 focus:ring-blue-400"
        />

        <button
          onClick={predictSentiment}
          className="mt-6 px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg font-semibold transition disabled:bg-gray-400"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict Sentiment"}
        </button>

        {/* Error */}
        <div className="mt-6 h-10">
          {error && (
            <h2 className=" text-xl font-semibold text-red-400">{error}</h2>
          )}

          {/* Success */}
          {!error && sentiment && (
            <h2 className=" text-xl font-semibold">Sentiment: {sentiment}</h2>
          )}
        </div>
      </div>
    </div>
  );
}
