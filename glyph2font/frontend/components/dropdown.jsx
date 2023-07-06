import Select from 'react-select';

const Dropdown = ({theme, options, setLanguage, language}) => {
    const mainTheme=theme;
    var value={}
    if (typeof language !== 'string'){
      value={value: 'en', label:'EN'}
    }
    else{
      value={value: language, label:language.toUpperCase()}
    }
    function handleChange (selectedOption){
      setLanguage(selectedOption.value);
    }
    return (
        <Select
          instanceId={1}
          options={options}
          value={value}
          onChange={handleChange}
          components={{
            IndicatorSeparator: () => null
          }}
          styles={{
            control: (baseStyles, state) => ({
              ...baseStyles,
              width: "90px",
              boxShadow: "none",
              borderRadius: 0,
              borderColor: state.isFocused ? mainTheme === "dark" ? "white" : "black" : mainTheme === "dark" ? "white" : "black", //button border color 
              background: mainTheme === "dark" ? "black" : "white", //button border color,
              '&:hover': {
                borderColor: mainTheme === "dark" ? "white" : "black",
              },
            }),
            singleValue: (provided, state) => ({
              ...provided,
              color: mainTheme === "dark" ? "white" : "black",
            }),
            menuList: (provided, state) => ({
              ...provided,
              paddingTop: 0,
              paddingBottom: 0,
            }),
            option: (provided, state) => ({
              ...provided,
              color: mainTheme === "dark" ? "white" : "black",
              backgroundColor: mainTheme === "dark" ? "black" : "white",
              '&:hover': {
                backgroundColor: mainTheme === "dark" ? "white" : "black",
                color: mainTheme === "dark" ? "black" : "white"
              }
            }),
          }}
        />
    );
};

export default Dropdown;