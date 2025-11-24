FROM node:18-alpine as build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Inject build-time environment variables for Vite
ARG VITE_BACKEND_URL
ARG VITE_MLFLOW_URL
ARG VITE_STREAMLIT_URL
ARG VITE_INTERFACE_URL
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
ENV VITE_MLFLOW_URL=${VITE_MLFLOW_URL}
ENV VITE_STREAMLIT_URL=${VITE_STREAMLIT_URL}
ENV VITE_INTERFACE_URL=${VITE_INTERFACE_URL}

RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Copy a custom nginx config if we need SPA routing support
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

