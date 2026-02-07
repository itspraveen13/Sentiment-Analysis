import React, { useEffect, useState } from 'react';
import { getStorage, ref, listAll, getDownloadURL } from "firebase/storage";
import {
    Box,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';
import Paper from '@mui/material/Paper';


const History = () => {
  const [historyData, setHistoryData] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(true); // Add loading state
  const loggedInUser = localStorage.getItem('username');

  useEffect(() => {
      // Fetch data from Firebase Storage
      const storage = getStorage();
      const storageRef = ref(storage, `data/${loggedInUser}/`);
      const splitFileName = (fileName) => {
          const parts = fileName.split('-');
          const date = parts[0];
          const time = parts[1];
          const file = parts.slice(2).join('-');
          return { date, time, file };
      };

      listAll(storageRef)
          .then(async (result) => {
              const dataPromises = result.items
                  .filter(itemRef => itemRef.name.endsWith('.xlsx'))
                  .map(async (itemRef) => {
                      const fileName = itemRef.name;
                      const { date, time, file } = splitFileName(fileName);

                      // Get the download URL for the file
                      const downloadURL = await getDownloadURL(itemRef);

                      return {
                          date: date,
                          time: time,
                          file,
                          downloadURL, 
                      };
                  });

              const data = await Promise.all(dataPromises);
              setHistoryData(data);
              setIsLoading(false); // Set loading to false once data is fetched
          })
          .catch((error) => {
              console.error('Error fetching data from Firebase Storage:', error);
              setIsLoading(false); // Set loading to false in case of error
          });
  }, [loggedInUser]);

  const handleOpenInWebsiteClick = (file) => {
      setSelectedFile(file);
  };
  
  return (
      <>
        <Box display="flex" justifyContent="center">
          {
            loggedInUser ? (
              <div id="histDiv" style={{  width: '80%', paddingBottom: '10%' }}>
            {isLoading ? (
              <h1 align='center'  style={{ fontWeight: 'normal' }}>Fetching user data, please wait...</h1>
            ) : historyData.length > 0 ? (
              <Box>
                <h2 align="center" style={{ fontSize: '30px', marginBottom: '3%' }}>Showing History of {loggedInUser}</h2>
                <TableContainer component={Paper} style={{  }}>
                <Table sx={{ minWidth: 650 }}>
                  <TableHead style={{ backgroundColor: '#404A86', border: '2px solid #404A86' }}>
                    <TableRow>
                      <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', padding: '10px', color: 'white' }}>S.No</TableCell>
                      <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', padding: '10px', color: 'white' }}>Date</TableCell>
                      <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', padding: '10px', color: 'white' }}>Time</TableCell>
                      <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', padding: '10px', color: 'white' }}>File Name</TableCell>
                      <TableCell align="center" style={{ fontSize: '20px', fontWeight: 'bold', padding: '10px', color: 'white' }}>Download Excel File</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody s>
                    {historyData.map((item, index) => (
                      <TableRow style={{ backgroundColor: 'white', border: '2px solid #404A86' }} key={index}>
                        <TableCell align="center" style={{ padding: '10px' }}>{index + 1}</TableCell>
                        <TableCell align="center" style={{ padding: '10px' }}>{item.date}</TableCell>
                        <TableCell align="center" style={{ padding: '10px' }}>{item.time}</TableCell>
                        <TableCell align="center" style={{ padding: '10px' }}>{item.file}</TableCell>
                        <TableCell align="center" style={{ padding: '10px' }}>
                          <a href={item.downloadURL} target="_blank" rel="noopener noreferrer">
                            Click here
                          </a>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              </Box>
            ) : (
              <h1 align='center' style={{ fontWeight: 'normal' }}>No data available to show!</h1>
            )}
          </div>
            ) : (
              <h1 align='center' style={{ fontWeight: 'normal' }}>Please Login!</h1>
            )
          }
        </Box>
      </>
  );
};

export default History;