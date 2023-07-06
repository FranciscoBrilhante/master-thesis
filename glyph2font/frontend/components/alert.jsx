import FeatherIcon from 'feather-icons-react';

const Alert = ({ visibility, message, handleSetVisibility, styles }) => {
    if (!visibility) return null;
    return (
        <div className={styles.alert}>
            <div className={styles.alertMessage}>{message}</div>
            <FeatherIcon icon="x" size="30" strokeWidth="1" className={styles.closeDown} onClick={(e) => { handleSetVisibility(false) }} />
        </div>
    );
};

export default Alert;