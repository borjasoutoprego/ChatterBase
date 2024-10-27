from langchain import HuggingFaceHub, LLMChain, PromptTemplate 
import os  
from psycopg2 import connect
from nl_to_sql import natural_language_to_sql
import sys
import streamlit as st

prompt_path = sys.argv[1]
prompt_sqlcoder_path = sys.argv[2]

with open(prompt_path, 'r') as template_file:
    template = template_file.read()

with open(prompt_sqlcoder_path, "r") as prompt_sqlcoder_file:
    prompt_sqlcoder = prompt_sqlcoder_file.read()

# User database
if 'users_db' not in st.session_state:
    st.session_state['users_db'] = {
        "admin": {"password": "admin", "role": "admin"},
        "user": {"password": "user", "role": "user"}
    }

# User authentication function
def authenticate_user(username, password):
    if username in st.session_state['users_db'] and st.session_state['users_db'][username]['password'] == password:
        return True, st.session_state['users_db'][username]['role']
    return False, None

# Feature for administrator to add new user
def add_user(username, password, role="user"):
    if username not in st.session_state['users_db']:
        st.session_state['users_db'][username] = {"password": password, "role": role}
        return True
    return False

# Function for administrator to delete a user
def remove_user(username):
    if username in st.session_state['users_db'] and st.session_state['users_db'][username]["role"] != "admin":
        del st.session_state['users_db'][username]
        return True
    return False

def connect_db(schema="raw"):
    """Connects to the PostgreSQL database server with an optional schema parameter"""

    conn = None
    try:
        conn = connect(
            host="localhost",
            dbname="supermarket",
            port=5432,
            user="postgres",
            password="postgres"
        )

        # Set the schema
        if schema:
            with conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema};")

        return conn

    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def get_db_connection():
    if "conn" not in st.session_state:
        st.session_state.conn = connect_db()
    return st.session_state.conn

def query_database(query):
    """Query the database and return the results"""

    conn = get_db_connection() 
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    return result

def generate_response(huggingfacehub_api_token, llm, human_input, template, temperature=0.1, top_p=0.9, top_k=50, max_new_tokens=256):
    """Generates a response to a user input"""

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            template += "User: " + dict_message["content"] + "\n\n"
        else:
            template += "Assistant: " + dict_message["content"] + "\n\n"

    # Convert the question to a SQL query
    sql_query = natural_language_to_sql(human_input, prompt_sqlcoder)
    # Get the results of the SQL query
    db_results = query_database(sql_query)
    # Convert the results to a string and remove the square brackets
    db_results = str(db_results).replace('[', '').replace(']', '')
    # Create the prompt with the question and the results of the SQL query
    prompt = PromptTemplate(
        template=template,
        input_variables=["human_input", "db_results"]
    )

    # Instantiate the language model with the prompt
    llm_chain = LLMChain(
        llm=HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, repo_id=llm, model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens, 
            "top_p": top_p, "top_k": top_k}),
        prompt=prompt,
        verbose=False,
    )

    # Get the response from the language model
    output = llm_chain.predict(human_input=human_input, db_results=db_results) 

    return output

def admin_panel():
    """Admin panel functionality to manage users"""
    st.sidebar.subheader("Admin Panel")
    new_username = st.sidebar.text_input("New user", key="new_username")
    new_password = st.sidebar.text_input("Password", type="password", key="new_password")
    if st.sidebar.button("Add user"):
        if add_user(new_username, new_password):
            st.sidebar.success(f"User {new_username} added successfully.")
        else:
            st.sidebar.error(f"User {new_username} already exists.")
    
    del_username = st.sidebar.text_input("User to delete", key="del_username")
    if st.sidebar.button("Delete user"):
        if remove_user(del_username):
            st.sidebar.success(f"User {del_username} successfully deleted.")
        else:
            st.sidebar.error("Error deleting user.")

def main():
    """Main function"""

    # App title
    st.set_page_config(page_title="💬 ChatterBase")

    with st.sidebar:
        st.title('💬 ChatterBase')

        # Request authentication
        username = st.sidebar.text_input("Username", key="username")
        password = st.sidebar.text_input("Password", type="password", key="password")
        authenticated, role = authenticate_user(username, password)

        if not authenticated and role is not None:
            st.error("Incorrect username or password.")

        huggingfacehub_api_token = None

        if authenticated:
            huggingfacehub_api_token = st.text_input('Enter Hugging Face API token:', type='password')
            if not (huggingfacehub_api_token.startswith('hf_') and len(huggingfacehub_api_token)==37):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('HUGGING FACE key provided!', icon='✅')
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token

            # Adjustment of model parameters
            st.subheader('Models and parameters')
            selected_model = st.sidebar.selectbox('Choose a model', ['Mixtral-8x7B-Instruct-v0.1', 'falcon-7b-instruct', 'Meta-Llama-3-8B-Instruct', 'gemma-2-2b-it'], key='selected_model')
            if selected_model == 'Mixtral-8x7B-Instruct-v0.1':
                llm = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            elif selected_model == 'falcon-7b-instruct':
                llm = 'tiiuae/falcon-7b-instruct'
            elif selected_model == 'Meta-Llama-3-8B-Instruct':
                llm = 'meta-llama/Meta-Llama-3-8B-Instruct'
            elif selected_model == 'gemma-2-2b-it':
                llm = 'google/gemma-2-2b-it'
            temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
            top_k = st.sidebar.slider('top_k', min_value=10, max_value=50, value=50, step=5)
            max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=32, max_value=512, value=256, step=32)

            if role == 'admin':
                admin_panel()

    if authenticated:

        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "¿What do you want to know about your data?"}]

        # Display or clear chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "¿What do you want to know about your data?"}]
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
                
        # User-provided prompt
        if human_input := st.chat_input(disabled=not huggingfacehub_api_token):
            st.session_state.messages.append({"role": "user", "content": human_input})
            with st.chat_message("user"):
                st.write(human_input)
        
        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Talking to your DataBase..."):
                    response = generate_response(huggingfacehub_api_token, llm, human_input, template, temperature, top_p, top_k, max_new_tokens)
                    last_line = response.strip().split('\n')[-1]
                    answer = last_line.split(": ", 1)[-1]
                    placeholder = st.empty()
                    full_response = ''
                    for item in answer:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == '__main__':
    main()