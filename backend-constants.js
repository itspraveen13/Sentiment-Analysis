// Backend Constants Configuration

export const SENTIMENT_CONFIG = {
  // Confidence score constants
  MIN_CONFIDENCE: parseFloat(process.env.MIN_CONFIDENCE_SCORE || "0.6"),
  MAX_CONFIDENCE: parseFloat(process.env.MAX_CONFIDENCE_SCORE || "0.99"),
  CONFIDENCE_THRESHOLD: parseFloat(process.env.CONFIDENCE_THRESHOLD || "0.5"),

  // Sentiment scoring
  POSITIVE_WEIGHT: 0.1,
  BASE_CONFIDENCE: 0.6,

  // Word lists for sentiment analysis
  POSITIVE_WORDS: [
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
    "wonderful",
    "outstanding",
    "superb",
    "terrific",
    "delighted",
    "brilliant",
  ],

  NEGATIVE_WORDS: [
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
    "pathetic",
    "disgusting",
    "dreadful",
    "mediocre",
    "inferior",
    "abysmal",
  ],
};

export const FILE_CONFIG = {
  MAX_FILE_SIZE: parseInt(process.env.MAX_FILE_SIZE || "10485760"), // 10MB
  MAX_ROWS_PER_ANALYSIS: parseInt(process.env.MAX_ROWS_PER_ANALYSIS || "100"),
  UPLOAD_DIR: "uploads",
  ALLOWED_FILE_TYPES: {
    csv: "text/csv",
    txt: "text/plain",
    audio: "audio/*",
  },
};

export const API_CONFIG = {
  PORT: process.env.BACKEND_PORT || 5000,
  NODE_ENV: process.env.NODE_ENV || "development",
  CORS_ORIGIN: process.env.CORS_ORIGIN || "*",
};

export const MESSAGES = {
  ERRORS: {
    NO_TEXT: "No text provided",
    NO_FILE: "No file uploaded",
    INVALID_FILE_TYPE: "Invalid file type",
    FILE_TOO_LARGE: "File size exceeds maximum limit",
    ANALYSIS_FAILED: "Error analyzing text",
    DATASET_FAILED: "Error analyzing dataset",
    AUDIO_FAILED: "Error analyzing audio",
    CHURN_FAILED: "Error predicting churn",
    FAKE_FAILED: "Error detecting fake reviews",
    SOCIAL_FAILED: "Error analyzing social media",
    MEME_FAILED: "Error analyzing meme",
  },
  SUCCESS: {
    ANALYSIS_COMPLETE: "Analysis completed successfully",
  },
};

export const ENDPOINTS = {
  TEXT: "/text",
  DATASET: "/dataset",
  AUDIO: "/audio",
  CHURN: "/churn",
  FAKE: "/fake",
  SOCIAL: "/social",
  MEME: "/meme",
  HEALTH: "/health",
};
