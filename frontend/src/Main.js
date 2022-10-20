import React, { useState, useEffect } from 'react';
import "./main.css";
import { IoMdCloudUpload } from "react-icons/io";
import axios from 'axios';


const Main = () => {
  const [image, setImage] = useState();
  const [file, setFile] = useState(false);
  const [data, setData] = useState();
  const [preview, setPreview] = useState();
  const [isLoading, setIsloading] = useState(false);

  useEffect(() => {
    if (!image) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(image);
    setPreview(objectUrl);
  }, [image]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    // setIsloading(true);
    // sendfile();
  }, [preview]);

  const fileHandler = (e) => {
    console.log(e.target.files[0]);
    setImage(e.target.files[0]);
    setFile(true);
  }



  const sendfile = async () => {
    let formData = new FormData();
    formData.append("file", image);
    let res = await axios({
      headers: { 'Access-Control-Allow-Origin': '*' },
      method: "post",
      url: 'http://localhost:8000/predict',
      data: formData,
      crossorigin: true
    });
    if (res.status === 200) {
      console.log(res.data);
      setData(res.data);
    }
    setIsloading(true);
  }
  return (
    <>
      <div>
        <div className="navbar">
          <h3 className="nav-text"> POTATO PLANT-DISEASE-DETECTOR</h3>
        </div>
        <div className="background-image"></div>
        <div className="box">
          <p className="text"> Upload a clear picture of the potato leaf </p>
          {file && <div className="upload-1">
            <img src={preview} alt="img" />
          </div>
          }
          {file ? "" : <div className="upload">
            <button className="btn-upload"><IoMdCloudUpload className="icon" size={25} />Choose file</button>
            <input type="file" onChange={(e) => fileHandler(e)} />
          </div>}

          <div className="btn-class">
            {
              isLoading ? <div className="res-text"><h3>Type-{data.class}</h3><h3>Accuracy-{(parseFloat(data.confidence) * 100).toFixed(2)}</h3></div> : <button className="btn-send" onClick={sendfile}>send</button>
            }

          </div>
        </div>
      </div>
    </>

  );
}

export default Main;