FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and rdkit (fixed)
RUN conda install python=3.10 rdkit -c conda-forge -y && conda clean -afy

# Copy requirements and install pip packages
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy all app files
COPY . .

# Expose port
EXPOSE 7860

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
