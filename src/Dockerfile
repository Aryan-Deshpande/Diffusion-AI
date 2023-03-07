FROM node:16

# creating a working directory /app
WORKDIR /app

# ensuring, package.json & package-lock.json are copied
COPY package*.json ./

# dependency production only installation
RUN npm ci --only=production 
COPY . .

EXPOSE 8080

CMD ["node", "Server.js"]