FROM tensorflow/serving

COPY /HutModule /ModelFolder/1

EXPOSE 8501

CMD ["tensorflow_model_server", \
    "--rest_api_port=8501", \
    "--model_name=HutModule", \ 
    "--model_base_path=/ModelFolder"]