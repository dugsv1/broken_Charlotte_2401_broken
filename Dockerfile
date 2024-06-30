FROM python:3.12-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements.txt early to leverage Docker cache
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install local package
RUN pip install -e .

# Installing requirements for wkhtmltox
RUN apt-get update && apt-get install -y \
    wget \
    xfonts-75dpi \
    xfonts-base \
    fontconfig \
    libjpeg62-turbo \
    xz-utils \
    libssl1.1  # Adding libssl1.1 which is required by wkhtmltox

# Download and install wkhtmltox
RUN wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.buster_amd64.deb \
    && dpkg -i wkhtmltox_0.12.6-1.buster_amd64.deb \
    && apt-get install -f  # Automatically fix broken dependencies \
    && rm wkhtmltox_0.12.6-1.buster_amd64.deb

# Update package list and install TeX Live
RUN apt-get update && apt-get install -y \
    texlive \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    latexmk

EXPOSE 8002

# For now, let's keep the container running with an idle command.
CMD ["sleep", "infinity"]
