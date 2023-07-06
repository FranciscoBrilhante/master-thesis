FROM node:18
RUN apt-get update -y
WORKDIR /home/frontend
COPY . .
RUN npm install next react react-dom -y
RUN npm install -y
EXPOSE 3000
CMD npm run dev

# docker build -t glyph2font-frontend .
# docker run -it -p 3000:3000 --name glyph2font-frontend glyph2font-frontend
# docker rm glyph2font-frontend

# npm run build
# npm run start