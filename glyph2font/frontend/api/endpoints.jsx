import config from "config.json";
const api_address = config.backend_address;

export async function modelsList() {
    const endpoint = "models"

    const payload = {
        method: "GET",
        credentials: "same-origin",
        headers: {
            "Content-Type": "application/json",
        },
    };

    try {
        const response = await fetch(api_address + endpoint, payload)
        if (!response.ok) {
            return { 'status': 400 };
        }
        else {
            return response.json()
        }
    } catch (error) {
        return { 'status': 400 };
    }
}



export async function generate(model, format, colorscheme, poligons, segments, labels, width, height, image) {
    const endpoint = "generate"
    var data=new FormData();
    data.append('model',model);
    data.append('format',format);
    data.append('colorscheme',colorscheme);
    data.append('poligons',poligons);
    data.append('segments',segments);
    data.append('labels',labels);
    data.append('width',width);
    data.append('height',height);
    data.append('images', image, "image.png");

    const payload = {
        method: "POST",
        credentials: "same-origin",
        body:data
    };

    try {
        const response = await fetch(api_address + endpoint, payload)
        if (!response.ok) {
            return { 'status': 400 };
        }
        else {
            return response.json()
        }
    } catch (error) {
        return { 'status': 400 };
    }
}