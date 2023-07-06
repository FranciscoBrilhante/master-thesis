import * as React from 'react';
import { styled } from '@mui/material/styles';
import Switch from '@mui/material/Switch';



const IOSSwitch = styled((props) => (
    <Switch focusVisibleClassName=".Mui-focusVisible" disableRipple {...props} />
))(({ theme, maintheme }) => ({
    width: 42,
    height: 26,
    padding: 0,
    '& .MuiSwitch-switchBase': {
        padding: 0,
        margin: 2,
        color: maintheme === 'dark' ? '#ffffff' : '#565656',
        transitionDuration: '300ms',
        '&.Mui-checked': {
            transform: 'translateX(16px)',
            color: '#ffffff',
            '& + .MuiSwitch-track': {
                backgroundColor:maintheme === 'dark' ? '#000000' : '#D7D7D7',
                opacity: 1,
                border: 0,
            },
            '&.Mui-disabled + .MuiSwitch-track': {
                opacity: 1,
            },
        },
        '&.Mui-focusVisible .MuiSwitch-thumb': {
            color: '#D7D7D7',
            border: '6px solid #fff',
        },
        '&.Mui-disabled .MuiSwitch-thumb': {
            color:
                theme.palette.mode === 'light'
                    ? "#D7D7D7"
                    : "#D7D7D7",
        },
        '&.Mui-disabled + .MuiSwitch-track': {
            opacity: theme.palette.mode === 'light' ? 1 : 1,
        },
    },
    '& .MuiSwitch-thumb': {
        boxSizing: 'border-box',
        width: 22,
        height: 22,
    },
    '& .MuiSwitch-track': {
        borderRadius: 26 / 2,
        backgroundColor:maintheme === 'dark' ? '#000000' : '#D7D7D7',
        opacity: 1,
        transition: theme.transitions.create(['background-color'], {
            duration: 500,
        }),
    },
}));


/*<IOSSwitch sx={{ m: 1 }} defaultChecked />*/
export default IOSSwitch;