FROM nginx:alpine

# Install envsubst for environment variable substitution
RUN apk add --no-cache gettext

# Create templates directory (nginx looks here for templates)
RUN mkdir -p /etc/nginx/templates

# Copy nginx configuration template
COPY infra/mlflow-proxy-railway.conf /etc/nginx/templates/default.conf.template

# Railway sets PORT automatically, but nginx listens on port 80 by default
EXPOSE 80

# Use envsubst to replace environment variables in nginx config
# Set MLFLOW_SERVICE_URL in Railway dashboard (e.g., http://mlflow-service.up.railway.app:5000)
# nginx will automatically process templates in /etc/nginx/templates/*.template
CMD ["sh", "-c", "envsubst '$$MLFLOW_SERVICE_URL' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]

