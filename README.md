# ChatterBase

Trabajo de Fin de Grado de Ciencia e Ingeniería de Datos desarrollado por Borja Souto, David Vilares y Bruno Cabado en la Universidad de A Coruña en 2024, bajo el título de "*Asistente virtual basado en grandes modelos de lenguaje para la interacción con bases de datos mediante lenguaje natural*".

![interfaz_tfg](https://github.com/user-attachments/assets/a9111c7e-bc9a-4b72-85fa-b447787678cc)

En este proyecto se implementa un `chatbot basado en LLMs` mediante el uso de las librerías [Hugging Face Transformers](https://huggingface.co/transformers/) y [Langchain](https://www.langchain.com/). Este chatbot permite la `interacción con una base de datos` PostgreSQL `mediante el uso de lenguaje natural`. Además, se desarrolla una interfaz de usuario con [Streamlit](https://www.streamlit.io/). La app se despliega directamente desde Google Colab con el uso de [Localtunnel](https://theboroer.github.io/localtunnel-www/).

## Archivos
- `dataset/supermarket_sales.csv`: [dataset utilizado](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales/data) (en formato .csv).
- `nl_to_sql.py`: script encargado de la traducción de una pregunta en lenguaje natural a una query SQL.
- `app.py`: script que contiene el funcionamiento principal de la aplicación.
- `exe.ipynb`: Jupyter notebook para la ejecución y despliegue del proyecto en Google Colab.
