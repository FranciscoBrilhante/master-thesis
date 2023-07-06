import { useState } from "react";

const HelpTab = ({ styles, locale, language, theme }) => {
    const [visible, setVisible] = useState(false);
    return (
        <div className={`${visible ? styles.helpTabContainer : styles.helpTabContainerHidden}`}>
            <div className={styles.helpTab}>
                <p>
                    <b>1.</b>  {locale["helpStep1"]} <br />
                    {locale["helpStep1Part2"]}
                </p>
                <p>
                    <b>2.</b>  {locale["helpStep2"]}
                </p>
                <p>
                    <b>3.</b>  {locale["helpStep3"]}
                </p>
                <p>
                    <b>4.</b> . {locale["helpStep4"]}
                </p>
                <p>
                    <b>5.</b>  {locale["helpStep5"]}
                </p>
                <p>
                    <b>6.</b>  {locale["helpStep6"]}
                </p>
                <p>
                    <b>7.</b>  {locale["helpStep7"]}
                </p>
            </div>
            <div className={styles.helpButton} onClick={(e) => { setVisible(!visible) }}>
                {locale["helpButton"]}
                
            </div>

        </div>
    );
};

export default HelpTab;