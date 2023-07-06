// next/react/node imports
import React, { useState, useRef, useEffect } from "react";
import FeatherIcon from 'feather-icons-react';
import Image from 'next/image'
// style sheets
import lightStyles from '@/styles/light/promptCard.module.css'
import darkStyles from '@/styles/dark/promptCard.module.css'

const PromptCard = ({label, addImages, theme}) => {
    const [styles, setStyles] = useState('light');
    useEffect(() => {
        setStyles(theme == 'dark' ? darkStyles : lightStyles);
    }, [theme]);

    function handleChange(e) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            
            var blob = e.target.files[0].slice(0, e.target.files[0].size, 'image/png'); 
            var newFile = new File([blob], label+'.png', {type: 'image/png'});
            addImages([newFile]);
        }
    };

    function handleClick(e) {
        e.target.value = null;
    }
    
    return (
        <div className={styles.container}>
            <div className={styles.image}>{label}</div>
            <input className={styles.fileUpload} type="file" id={"input-file-upload"+label} multiple={false} accept="image/jpeg, image/jpg, image/png" onChange={handleChange} onClick={handleClick} />
            <label className={styles.fileUploadLabel} htmlFor={"input-file-upload"+label}>
                <FeatherIcon icon="upload" size="20" strokeWidth="1" />
                <div className={styles.addPrompt}>upload</div>
            </label>

        </div>
    );
};

export default PromptCard;