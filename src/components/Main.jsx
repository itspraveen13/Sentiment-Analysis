import { Box, Stack } from "@mui/material";
import React from "react";
import EmojiEmotionsIcon from "@mui/icons-material/EmojiEmotions";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import ConnectWithoutContactIcon from "@mui/icons-material/ConnectWithoutContact";
import DocumentScannerIcon from "@mui/icons-material/DocumentScanner";
import "../assets/css/home.css";
import icon from "../assets/img/meme.png";
const Main = () => {
  return (
    <>
      <div id="htitle">
        <h1>
          Get Insights Using <br /> Sentiment Analysis{" "}
        </h1>
        <h2>Understand your customers like never before!</h2>
      </div>
      <Stack
        sx={{ margin: "0px 10%", flexDirection: { sm: "row", xs: "column" } }}
      >
        <a href="/sen">
          <div id="card" className="c1">
            <div id="icon">
              <EmojiEmotionsIcon sx={{ fontSize: "50px" }} />
            </div>
            <h3>Sentiment Analysis </h3>
          </div>
        </a>
        <a href="/cp">
          <div id="card" className="c2">
            <div id="icon">
              <AutoGraphIcon sx={{ fontSize: "50px" }} />
            </div>
            <h3>Churn Prediction </h3>
          </div>
        </a>
        <a href="/social">
          <div id="card" className="c3">
            <div id="icon">
              <ConnectWithoutContactIcon sx={{ fontSize: "50px" }} />
            </div>
            <h3>Social Listening </h3>
          </div>
        </a>
        <a href="/fake">
          <div id="card" className="c4">
            <div id="icon">
              <DocumentScannerIcon sx={{ fontSize: "50px" }} />
            </div>
            <h3>Fake Detection </h3>
          </div>
        </a>
        <a href="/dash">
          <div id="card" className="c4">
            <div id="icon">
              <img
                src={icon}
                alt=""
                style={{ height: "40px", width: "40px", marginRight: "10%" }}
              />
            </div>
            <h3>Meme Analysis </h3>
          </div>
        </a>
      </Stack>
    </>
  );
};

export default Main;
