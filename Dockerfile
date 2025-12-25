FROM continuumio/miniconda3:latest

WORKDIR /app

# Install conda packages
RUN conda install -c conda-forge rdkit numpy pandas -y && \
    conda clean -afy

# Install pip packages
RUN pip install --no-cache-dir streamlit torch torch-geometric

# Copy app files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
