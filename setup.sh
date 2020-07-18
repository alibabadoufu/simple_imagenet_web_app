mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"woenyon001@e.ntu.edu.sg\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\