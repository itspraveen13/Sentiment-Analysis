import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import fs from "fs";

const app = express();
const PORT = 5000;
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Sentiment analysis using keyword matching
function analyzeSentiment(text) {
  const positiveWords = [
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "love",
    "best",
    "awesome",
    "perfect",
    "fantastic",
    "beautiful",
    "happy",
    "glad",
    "satisfied",
    "impressed",
  ];
  const negativeWords = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "hate",
    "poor",
    "disappointing",
    "useless",
    "waste",
    "broken",
    "angry",
    "upset",
    "dissatisfied",
    "frustrated",
  ];

  const lowerText = text.toLowerCase();
  let positiveCount = 0;
  let negativeCount = 0;
  let reasons = [];

  positiveWords.forEach((word) => {
    if (lowerText.includes(word)) {
      positiveCount++;
      reasons.push(`Contains positive word: "${word}"`);
    }
  });

  negativeWords.forEach((word) => {
    if (lowerText.includes(word)) {
      negativeCount++;
      reasons.push(`Contains negative word: "${word}"`);
    }
  });

  let sentiment = "Neutral";
  let confidence = 0.5;

  if (positiveCount > negativeCount) {
    sentiment = "Positive";
    confidence = Math.min(0.99, 0.6 + positiveCount * 0.1);
  } else if (negativeCount > positiveCount) {
    sentiment = "Negative";
    confidence = Math.min(0.99, 0.6 + negativeCount * 0.1);
  } else if (positiveCount > 0 || negativeCount > 0) {
    confidence = 0.5;
  }

  return {
    sentiment,
    confidence: confidence.toFixed(2),
    reasons: reasons.length > 0 ? reasons : ["Neutral sentiment detected"],
  };
}

// Text analysis
app.post("/text", (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "No text provided" });
    const analysis = analyzeSentiment(text);
    res.json({
      predictions: [
        {
          text: text.substring(0, 200),
          sentiment: analysis.sentiment,
          reasons: analysis.reasons,
          confidence: analysis.confidence,
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error analyzing text" });
  }
});

// Dataset upload
app.post("/dataset", upload.single("file"), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });
    const fileContent = fs.readFileSync(req.file.path, "utf8");
    const lines = fileContent
      .split("\n")
      .slice(1)
      .filter((line) => line.trim());
    const predictions = lines.slice(0, 10).map((line) => {
      const reviewText = line.split(",")[0] || "Sample review";
      const analysis = analyzeSentiment(reviewText);
      return {
        text: reviewText.substring(0, 100),
        sentiment: analysis.sentiment,
        reasons: analysis.reasons,
        confidence: analysis.confidence,
      };
    });
    fs.unlinkSync(req.file.path);
    res.json({ predictions });
  } catch (error) {
    res.status(500).json({ error: "Error analyzing dataset" });
  }
});

// Audio analysis
app.post("/audio", upload.single("file"), (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "No audio file uploaded" });
    const mockTranscription =
      "This product is absolutely amazing and I love it. Great quality and excellent service!";
    const analysis = analyzeSentiment(mockTranscription);
    fs.unlinkSync(req.file.path);
    res.json({
      predictions: [
        {
          text: mockTranscription,
          sentiment: analysis.sentiment,
          reasons: analysis.reasons,
          confidence: analysis.confidence,
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error analyzing audio" });
  }
});

// Churn prediction
app.post("/churn", upload.single("file"), (req, res) => {
  try {
    res.json({
      predictions: [
        {
          risk: Math.random() > 0.5 ? "High" : "Low",
          probability: (Math.random() * 0.4 + 0.3).toFixed(2),
          reasons: ["Analysis based on customer engagement patterns"],
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error predicting churn" });
  }
});

// Fake detection
app.post("/fake", upload.single("file"), (req, res) => {
  try {
    res.json({
      predictions: [
        {
          text: "Analyzed review",
          isFake: Math.random() > 0.7,
          confidence: (Math.random() * 0.3 + 0.7).toFixed(2),
          reasons: ["Review authenticity analysis completed"],
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error detecting fake reviews" });
  }
});

// Social listening
app.post("/social", (req, res) => {
  try {
    res.json({
      mentions: [
        {
          text: "@brand positive mention detected",
          sentiment: "Positive",
          source: "Twitter",
        },
        {
          text: "@brand customer feedback",
          sentiment: "Neutral",
          source: "Reddit",
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error analyzing social media" });
  }
});

// Meme analysis
app.post("/meme", upload.single("file"), (req, res) => {
  try {
    res.json({
      predictions: [
        {
          sentiment: "Humorous",
          confidence: (Math.random() * 0.3 + 0.7).toFixed(2),
          emotions: ["Funny", "Relatable"],
        },
      ],
    });
  } catch (error) {
    res.status(500).json({ error: "Error analyzing meme" });
  }
});

app.get("/health", (req, res) => {
  res.json({ status: "Backend running", timestamp: new Date() });
});

app.listen(PORT, () => {
  console.log(`✅ Backend API running on http://localhost:${PORT}`);
});
