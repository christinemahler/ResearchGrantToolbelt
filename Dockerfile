# Get a distribution that has uv already installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Add user - this is the user that will run the app
# If you do not set user, the app will run as root (undesirable)
RUN useradd -m -u 1000 user
USER user

# 🔹 Install required system packages
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    lmodern \
    ghostscript \
    wkhtmltopdf \
    xvfb \
    && apt-get clean

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets

WORKDIR $HOME/app

# Set the working directory
WORKDIR $HOME/app

# Copy the app to the container
COPY --chown=user . $HOME/app

# Install the dependencies
# RUN uv sync --frozen
RUN uv sync

# Copy data
COPY ./data/Opportunities ~/app/data/Opportunities

# Expose the port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
