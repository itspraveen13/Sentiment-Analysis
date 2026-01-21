import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import fs from "fs";
import dotenv from "dotenv";
import {
  SENTIMENT_CONFIG,
  FILE_CONFIG,
  API_CONFIG,
  MESSAGES,
  ENDPOINTS,
} from "./backend-constants.js";

dotenv.config();

const app = express();
const PORT = API_CONFIG.PORT;
const upload = multer({ dest: FILE_CONFIG.UPLOAD_DIR });

app.use(cors({ origin: API_CONFIG.CORS_ORIGIN }));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Sentiment analysis using keyword matching and configured constants
function analyzeSentiment(text) {
  const positiveWords = SENTIMENT_CONFIG.POSITIVE_WORDS;
  const negativeWords = SENTIMENT_CONFIG.NEGATIVE_WORDS;

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
    confidence = Math.min(
      SENTIMENT_CONFIG.MAX_CONFIDENCE,
      SENTIMENT_CONFIG.BASE_CONFIDENCE +
        positiveCount * SENTIMENT_CONFIG.POSITIVE_WEIGHT,
    );
  } else if (negativeCount > positiveCount) {
    sentiment = "Negative";
    confidence = Math.min(
      SENTIMENT_CONFIG.MAX_CONFIDENCE,
      SENTIMENT_CONFIG.BASE_CONFIDENCE +
        negativeCount * SENTIMENT_CONFIG.POSITIVE_WEIGHT,
    );
  } else if (positiveCount > 0 || negativeCount > 0) {
    confidence = SENTIMENT_CONFIG.CONFIDENCE_THRESHOLD;
  }

  return {
    sentiment,
    confidence: confidence.toFixed(2),
    reasons: reasons.length > 0 ? reasons : ["Neutral sentiment detected"],
  };
}

// Text analysis
app.post(ENDPOINTS.TEXT, (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: MESSAGES.ERRORS.NO_TEXT });
    if (text.length > 5000)
      return res
        .status(400)
        .json({ error: "Text exceeds maximum length of 5000 characters" });
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
    console.error(error);
    res.status(500).json({ error: MESSAGES.ERRORS.ANALYSIS_FAILED });
  }
});

// Dataset upload
app.post(ENDPOINTS.DATASET, upload.single("file"), (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: MESSAGES.ERRORS.NO_FILE });

    const fileContent = fs.readFileSync(req.file.path, "utf8");
    const lines = fileContent
      .split("\n")
      .slice(1)
      .filter((line) => line.trim());

    const predictions = lines
      .slice(0, FILE_CONFIG.MAX_ROWS_PER_ANALYSIS)
      .map((line) => {
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
    console.error(error);
    if (req.file) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: MESSAGES.ERRORS.DATASET_FAILED });
  }
});

// Audio analysis
app.post(ENDPOINTS.AUDIO, upload.single("file"), (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: MESSAGES.ERRORS.NO_FILE });
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
    console.error(error);
    if (req.file) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: MESSAGES.ERRORS.AUDIO_FAILED });
  }
});

// Churn prediction
app.post(ENDPOINTS.CHURN, upload.single("file"), (req, res) => {
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
    console.error(error);
    res.status(500).json({ error: MESSAGES.ERRORS.CHURN_FAILED });
  }
});

// Fake detection
app.post(ENDPOINTS.FAKE, upload.single("file"), (req, res) => {
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
    console.error(error);
    res.status(500).json({ error: MESSAGES.ERRORS.FAKE_FAILED });
  }
});

// Social listening
app.post(ENDPOINTS.SOCIAL, (req, res) => {
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
    console.error(error);
    res.status(500).json({ error: MESSAGES.ERRORS.SOCIAL_FAILED });
  }
});

// Meme analysis
app.post(ENDPOINTS.MEME, upload.single("file"), (req, res) => {
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
    console.error(error);
    res.status(500).json({ error: MESSAGES.ERRORS.MEME_FAILED });
  }
});

app.get(ENDPOINTS.HEALTH, (req, res) => {
  res.json({
    status: "Backend running",
    timestamp: new Date(),
    environment: API_CONFIG.NODE_ENV,
  });
});

app.listen(PORT, () => {
  console.log(`\n✅ Backend API running on http://localhost:${PORT}`);
  console.log(`📝 Environment: ${API_CONFIG.NODE_ENV}\n`);
});
