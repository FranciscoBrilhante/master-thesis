import Image from 'next/image'

const VerticalSeparator = ({styles }) => {
    return (
        <div className={styles.separatorContainer}>
            <Image
                className={styles.separator}
                src={"/images/separator.svg"}
                alt={"panel separator line"}
                fill
            />
        </div>
    );
}

export default VerticalSeparator