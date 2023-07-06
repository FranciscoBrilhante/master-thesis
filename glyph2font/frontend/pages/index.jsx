// next/react/node imports
import Head from 'next/head'
import NextImage from 'next/image'
import React, { useState, useRef, useEffect } from "react";
import FeatherIcon from 'feather-icons-react';
import mergeImages from 'merge-images';
// fonts
import { Inter } from 'next/font/google'
// components
import VerticalSeparator from "components/verticalSeparator";
import IOSSwitch from "components/switch";
import Alert from "components/alert";
import LoadingDots from "components/loadingDots";
import GlyphCard from "components/glyphCard";
import PromptCard from "components/promptCard";
import Dropdown from "components/dropdown";
import HelpTab from "components/helpTab";
// configs/endpoints
import { modelsList, generate } from "api/endpoints";
import config from "config.json";
// style sheets
import lightStyles from '@/styles/light/index.module.css'
import darkStyles from '@/styles/dark/index.module.css'
//locales
import en from "locales/en.json";
import pt from "locales/pt.json";

const inter = Inter({ subsets: ['latin'] })

const Home = ({ s }) => {
  const [theme, setTheme] = useState(null);
  const [language, setLanguage] = useState(null);
  const [images, setImages] = useState([]);
  const [labels, setLabels] = useState([]);
  const [suggestedInputs, setSuggestedInputs] = useState(null);
  const [lastDimensions, setLastDimensions] = useState(null);

  function addImages(files) {
    var newImages = [...images];
    for (const file of files) {
      const newURL = URL.createObjectURL(file);
      newImages.push(newURL);
      const img = new Image();
      img.src = newURL;
      img.onload = () => {
        console.log(img.naturalHeight);
        console.log(img.naturalWidth);
        setLastDimensions({
          height: img.naturalHeight,
          width: img.naturalWidth
        });
      };
    }
    var newLabels = [...labels];
    for (const file of files) {
      var name = file.name.split(/(\\|\/)/g).pop().replace(/\.[^/.]+$/, "");
      newLabels.push(name);
    }

    setImages(newImages);
    setLabels(newLabels);
  }

  function deleteImage(index) {
    var newImages = [...images];
    newImages.splice(index, 1);
    setImages(newImages);

    var newLabels = [...labels];
    newLabels.splice(index, 1);
    setLabels(newLabels);
  }

  function changeLabel(index, label) {
    var newLabels = [...labels];
    newLabels[index] = label;
    setLabels(newLabels);
  }
  useEffect(() => {
    var themePreference = localStorage.getItem("theme") || "light";
    setTheme(themePreference);
    var languagePreference = localStorage.getItem("language") || "en";
    setLanguage(languagePreference);
  }, []);

  if (language == null || theme == null) { return; }
  var styles = theme == 'dark' ? darkStyles : lightStyles;
  var locale = language == 'en' ? en : pt;

  return (
    <>
      <Head>
        <title>Glyph2Font</title>
        <meta name="description" content="Create unique fonts with the power of AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        
        <link rel="apple-touch-icon" sizes="180x180" href="images/favicon/apple-touch-icon.png"/>
        <link rel="icon" type="image/png" sizes="32x32" href="images/favicon/favicon-32x32.png"/>
        <link rel="icon" type="image/png" sizes="16x16" href="images/favicon/favicon-16x16.png"/>
      </Head>
      <main className={styles.main}>
        <Header styles={styles} locale={locale} language={language} setLanguage={setLanguage} theme={theme} setTheme={setTheme}></Header>
        <Canvas styles={styles} locale={locale} theme={theme} images={images} labels={labels} addImages={addImages} deleteImage={deleteImage} changeLabel={changeLabel} suggestedInputs={suggestedInputs}></Canvas>
        <ControlPanel styles={styles} locale={locale} language={language} theme={theme} images={images} labels={labels} lastDimensions={lastDimensions} setSuggestedInputs={setSuggestedInputs} setImages={setImages} setLabels={setLabels}></ControlPanel>
        <HelpTab styles={styles} locale={locale} language={language} theme={theme}></HelpTab>
      </main>
    </>
  )
}

const Header = ({ styles, theme, setTheme, locale, language, setLanguage }) => {
  const options = [
    { value: 'en', label: 'EN' },
    { value: 'pt', label: 'PT' },
  ]
  function changeTheme(theme) {
    localStorage.setItem("theme", theme);
    setTheme(theme);
  }

  function changeLanguage(language) {
    localStorage.setItem("language", language);
    setLanguage(language);
  }

  return (
    <div className={styles.header}>
      <div className={styles.title}>
        Glyph2Font
      </div>
      <div className={styles.searchbar}></div>
      <div className={styles.options}>
        <FeatherIcon
          icon={theme == "dark" ? "sun" : "moon"}
          onClick={(e) => { changeTheme(theme == "dark" ? "light" : "dark") }}
          className={styles.themeButton}
        />
        <Dropdown
          setLanguage={changeLanguage}
          language={language}
          theme={theme}
          options={options}
        />
      </div>
    </div>
  );
}

const Canvas = ({ styles, theme, images, labels, addImages, deleteImage, changeLabel, suggestedInputs, locale }) => {
  const [dragActive, setDragActive] = useState(false);
  function handleDrag(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      addImages(e.dataTransfer.files);
    }
  };

  function handleChange(e) {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      addImages(e.target.files);
    }
  };

  function handleClick(e) {
    e.target.value = null;
  }

  if (images.length == 0 && suggestedInputs == null) {
    return (
      <form className={styles.mainFileUploadContainer} onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
        <input type="file" id="input-file-upload" multiple={true} className={styles.mainFileUpload} accept="image/jpeg, image/jpg, image/png" onChange={handleChange} onClick={handleClick} />
        <label htmlFor="input-file-upload" className={styles.mainFileUploadLabel}>
          {dragActive && <div className={styles.dragAreaOverlay} onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div>}
          <div className={styles.addPrompt}>{locale["mainUploadPrompt1"]}<br />{locale["mainUploadPrompt2"]}<br />{locale["mainUploadPrompt3"]}</div>
          <div className={styles.dropInArrowContainer}>
            <NextImage
              className={styles.dropInArrow}
              src={theme == "dark" ? "/images/dropInArrowDark.svg" : "/images/dropInArrow.svg"}
              alt={"drop letters here indicative arrow"}
              fill
            />
          </div>
        </label>
      </form>
    );
  }

  if (images.length != 0 && suggestedInputs == null) {
    return (
      <form className={styles.secondFileUploadContainer} onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
        <div className={styles.canvasFilled}>
          {images.map((image, i) => {
            return <GlyphCard
              theme={theme}
              key={i}
              index={i}
              deleteImageHandler={deleteImage}
              changeLabelHandler={changeLabel}
              image={image}
              label={labels[i]}
            />
          }
          )}
          <input type="file" id="input-file-upload" multiple={true} className={styles.secondFileUpload} accept="image/jpeg, image/jpg, image/png" onChange={handleChange} onClick={handleClick} />
          <label htmlFor="input-file-upload" className={styles.secondFileUploadLabel}>
            {dragActive && <div className={styles.dragAreaOverlay} onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div>}
            <FeatherIcon icon="plus" size="50" strokeWidth="1" />
            <div className={styles.secondAddPrompt}>{(locale["secondUploadPrompt1"])} <br /> {(locale["secondUploadPrompt2"])}</div>
          </label>
        </div>
      </form>
    );
  }

  return (
    <form className={styles.secondFileUploadContainer} onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
      <div className={styles.canvasFilled}>
        {suggestedInputs.split("").map((elem, i) => {
          if (labels.indexOf(elem) != -1) {
            return <GlyphCard
              theme={theme}
              key={i}
              index={labels.indexOf(elem)}
              deleteImageHandler={deleteImage}
              changeLabelHandler={changeLabel}
              image={images[labels.indexOf(elem)]}
              label={labels[labels.indexOf(elem)]}
            />
          }
          else {
            return <PromptCard
              theme={theme}
              key={i}
              label={elem}
              addImages={addImages}
            />
          }
        })}
        {images.map((image, i) => {
          if (suggestedInputs.split("").includes(labels[i])) {
            return null;
          }
          else {
            return <GlyphCard
              theme={theme}
              key={i}
              index={i}
              deleteImageHandler={deleteImage}
              changeLabelHandler={changeLabel}
              image={image}
              label={labels[i]}
            />
          }
        })}
        <input type="file" id="input-file-upload" multiple={true} className={styles.secondFileUpload} accept="image/jpeg, image/jpg, image/png" onChange={handleChange} onClick={handleClick} />
        <label htmlFor="input-file-upload" className={styles.secondFileUploadLabel}>
          {dragActive && <div className={styles.dragAreaOverlay} onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div>}
          <FeatherIcon icon="plus" size="50" strokeWidth="1" />
          <div className={styles.secondAddPrompt}>
            {(locale["secondUploadPrompt1"])} <br /> {(locale["secondUploadPrompt2"])}
          </div>
        </label>
      </div>
    </form>
  );

}

const ControlPanel = ({ styles, theme, images, labels, lastDimensions, setSuggestedInputs, setImages, setLabels, locale, language }) => {
  const [checked, setChecked] = useState(false)
  const [poligons, setPoligons] = useState(4);
  const [segments, setSegments] = useState(6);
  const [model, setModel] = useState(-1);
  const [scheme, setScheme] = useState('black');
  const [models, setModels] = useState([]);
  const [fontpath, setFontPath] = useState(null);
  const [alertVisible, setAlertVisible] = useState(false);
  const [alertOptions, setAlertOptions] = useState({ message: "", messageType: 0, });
  const [loading, setLoading] = useState(false);
  const [moreInfo, setMoreInfo] = useState(false);
  const [eta, setEta] = useState(-1);
  useEffect(() => {
    modelsList().then((response) => {
      if (response['status'] === 200) {
        var newModels = new Array();
        setModel(0);
        setSuggestedInputs(response["models"][0]["inputs"]);
        response['models'].map((model, i) => {
          newModels.push(model);
        })
        setModels(newModels);
      }
      else {
        setAlertOptions({ 'message': locale["modelsUnavailable"], 'messageType': 0 });
        setAlertVisible(true);
        setModels(new Array());
      }
    });
  }, []);

  function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
      byteString = atob(dataURI.split(',')[1]);
    else
      byteString = unescape(dataURI.split(',')[1]);
    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ia], { type: mimeString });
  }

  function generateFont() {
    if (model == -1) {
      setAlertOptions({ 'message': locale["noModelSelected"], 'messageType': 0 });
      setAlertVisible(true);
      return;
    }
    if (images.length == 0) {
      setAlertOptions({ 'message': locale["noInputs"], 'messageType': 0 });
      setAlertVisible(true);
      return;
    }
    setLoading(true);
    var setup = [];
    images.map((image, i) => {
      setup.push({ src: image, x: lastDimensions.width * i, y: 0 })
    })
    mergeImages(setup, { width: lastDimensions.width * images.length, height: lastDimensions.height, format: 'image/png' }).then(async b64 => {
      var aux = "";
      labels.map((letter, i) => {
        aux += letter;
      })
      generate(models[model]['name'], checked ? 'svg' : 'png', scheme, poligons, segments, aux, lastDimensions.width, lastDimensions.height, dataURItoBlob(b64)).then((response) => {
        setLoading(false);
        if (response.status === 200) {
          setFontPath(config.backend_address + response.font);
        }
        else {
          setAlertOptions({ 'message': locale["errorOcurred"], 'messageType': 0 });
          setAlertVisible(true);
        }
      })
    })

  }

  function resetAll() {
    setFontPath(null);
    setChecked(false);
    setPoligons(4);
    setSegments(6);
    setScheme("black");
    setImages([]);
    setLabels([]);
  }

  useEffect(() => {
    if (model == -1) {
      setEta("__");
      return;
    }

    let total = models[model].eta;
    if (checked) {
      total += 26 * poligons * 5;
    }
    const minutes = Math.floor(total / 60);
    const seconds = total - minutes * 60;

    var verbose = minutes + "min " + seconds + "sec";
    setEta(verbose);
  }, [model, poligons, segments, checked, models]);

  return (
    <div className={styles.controlPanel}>
      <Alert
        visibility={alertVisible}
        message={alertOptions.message}
        messageType={alertOptions.messageType}
        handleSetVisibility={(value) => setAlertVisible(value)}
        styles={styles}
      ></Alert>
      <div className={styles.panel}>
        <div className={styles.colorPanelTitle}>
          {(locale["colorSchemeTitle"])}
          <div className={styles.moreInfoTitle}>
          </div>
        </div>
        <div className={styles.colorSchemes}>
          <div className={`${scheme === 'black' ? styles.schemeChecked : styles.scheme}`} onClick={(e) => { setScheme('black') }}>
            <div className={styles.frameContainer}>
              <NextImage
                className={styles.frame}
                src={"/images/whiteFrame.svg"}
                alt={"white frame with dark letters"}
                fill
              />
            </div>
            <div className={styles.name}>{(locale["darkScheme"])}</div>
          </div>
          <div className={`${scheme === 'white' ? styles.schemeChecked : styles.scheme}`} onClick={(e) => { setScheme('white') }}>
            <div className={styles.frameContainer}>
              <NextImage
                className={styles.frame}
                src={"/images/blackFrame.svg"}
                alt={"white frame with dark letters"}
                fill
              />
            </div>
            <div className={styles.name}>{(locale["lightScheme"])}</div>
          </div>
        </div>
      </div>
      <VerticalSeparator styles={styles} />
      <div className={styles.panel}>
        <div className={styles.switchContainer}>
          {(locale["formatTitle"])}
        </div>
        <div className={styles.switchContainer}>
          PNG
          <IOSSwitch
            sx={{ m: 1 }}
            checked={checked}
            maintheme={theme}
            onChange={(e) => { setChecked(e.target.checked) }} />
          SVG
          {checked &&
            <div className={styles.svgMoreInfo} onClick={(e) => { setMoreInfo(!moreInfo) }}>
              <FeatherIcon icon="info" size="15" strokeWidth="2" />
              {moreInfo && <div className={styles.moreInfoText}>
                {(locale["formatMoreInfo1"])}<p />{(locale["formatMoreInfo2"])}</div>}
            </div>}
        </div>
        {checked &&
          <div className={styles.inputs}>
            <div className={styles.inputContainer}>
              <div className={styles.inputLabel}>{(locale["poligonsInput"])}</div>
              <div className={styles.input}><input className={styles.input} type="number" value={poligons} min="1" max="20" onChange={(e) => setPoligons(e.target.value)}></input></div>
            </div>
            <div className={styles.inputContainer}>
              <div className={styles.inputLabel}>{(locale["segmentsInput"])}</div>
              <div className={styles.input}><input className={styles.input} type="number" value={segments} min="1" max="20" onChange={(e) => setSegments(e.target.value)}></input ></div>
            </div>
          </div>}
      </div>
      <VerticalSeparator styles={styles} />
      <div className={styles.horizontalPanel}>
        <div className={styles.modelsContainer}>
          <div className={styles.modelsTitle}>{(locale["modelTitle"])}</div>
          <div className={styles.models}>
            {models.map((currModel, i) => {
              return (
                <div key={i} className={`${model === i ? styles.modelChecked : styles.model}`} onClick={(e) => { setModel(i); setSuggestedInputs(currModel["inputs"]) }} title={models[i]["description_pt"]}>{currModel["name"]} </div>
              );
            })}
          </div>
        </div>
        <div className={styles.generateContainer}>
          <div className={styles.etaContainer} onClick={(e) => { resetAll(); }}>
            <b>ETA:&nbsp;</b>{eta}
          </div>
          {loading == true
            ?
            <div className={styles.generateButton}>
              <LoadingDots className={styles.bouncingLoader} />
            </div>
            : (fontpath == null
              ?
              <div className={styles.generateButton} onClick={(e) => { generateFont() }}>{(locale["generateButton"])}</div>
              :
              <a href={fontpath} className={styles.openOutput} download>
                {(locale["openFontButton"])}
              </a>)}

          <div className={styles.resetContainer} onClick={(e) => { resetAll(); }}>
            <FeatherIcon icon="x" size="15" strokeWidth="2" />
            {(locale["clearAllButton"])}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
