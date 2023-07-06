// next/react/node imports
import React, { useState, useRef, useEffect } from "react";
import FeatherIcon from 'feather-icons-react';
import Image from 'next/image'
// style sheets
import lightStyles from '@/styles/light/glyphCard.module.css'
import darkStyles from '@/styles/dark/glyphCard.module.css'

const GlyphCard = ({ index, deleteImageHandler, changeLabelHandler, image, label, theme }) => {
    const [styles, setStyles] = useState('light');
    useEffect(() => {
        setStyles(theme == 'dark' ? darkStyles : lightStyles);
    }, [theme]);

    return (
        <div className={styles.letterPreviewContainer}>
            <div className={styles.deletePreview} onClick={(e) => { deleteImageHandler(index) }}><FeatherIcon icon="x" size="25" strokeWidth="1" /></div>
            <input className={styles.labelPreview} value={label} onChange={(e) => { changeLabelHandler(index, e.target.value) }}></input >
            <Image
                className={styles.letterPreview}
                src={image}
                alt={"preview of a character image"}
                fill
            />
        </div>
    );
};

export default GlyphCard;