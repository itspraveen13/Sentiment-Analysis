import Navbar from './component/Navbar'
import Main from './component/Main.jsx'
import Features from './component/Features'
import { Route, Routes } from 'react-router-dom'
import Sentiment from './component/Sentiment.jsx'
import Churn from './component/Churn.jsx'
import Social from './component/Social.jsx'
import Fake from './component/Fake.jsx'
import Meme from './component/Meme.jsx'
import About from './component/About.jsx'
import History from './component/History.jsx'

function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path='/' element={<Main />} />
        <Route path='/feature' element={<Features />} />
        <Route path='/sentiment' element={<Sentiment />} />
        <Route path='/churn' element={<Churn />} />
        <Route path='/social' element={<Social />} />
        <Route path='/fake' element={<Fake />} />
        <Route path='/meme' element={<Meme />} />
        <Route path='/about' element={<About/>} />
        <Route path='/history' element={<History/>} />
      </Routes>
    </>
  )
}

export default App
