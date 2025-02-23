FROM apache/airflow:2.7.2

WORKDIR /opt/airflow

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the dags and src folders
COPY ./dags /opt/airflow/dags
COPY ./src /opt/airflow/src

# Add src to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"








