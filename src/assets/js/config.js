import { initializeApp } from "firebase/app";
import { getAuth } from 'firebase/auth'
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyAeX2wFkCMXFMRMvkHuSgZYmzTif62X1MM",
  authDomain: "sentimetryx.firebaseapp.com",
  projectId: "sentimetryx",
  storageBucket: "sentimetryx.appspot.com",
  messagingSenderId: "876549296005",
  appId: "1:876549296005:web:34352cd9c3b42183dcfeec",
  measurementId: "G-WWXBPKLTRS"
};

const app = initializeApp(firebaseConfig);
export const database = getAuth(app);
export const storage = getStorage(app);