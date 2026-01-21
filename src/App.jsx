import { Route, Routes } from "react-router"
import Home from "./pages/Home"
import SentimentAnalysis from "./pages/SentimentAnalysis"
import SocialMedia from "./pages/SocialMedia"
import ChurnPrediction from "./pages/ChurnPrediction"
import FakeAnalysis from "./pages/FakeAnalysis"
import Dashboard from "./pages/Dashboard"
import About from "./pages/About"

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sen" element={<SentimentAnalysis />} />
        <Route path="/social" element={<SocialMedia />} />
        <Route path="/cp" element={<ChurnPrediction />} />
        <Route path="/fake" element={<FakeAnalysis />} />
        <Route path="/dash" element={<Dashboard />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </>
  )
}

export default App
