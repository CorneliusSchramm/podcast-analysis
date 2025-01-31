mkdir -p ~/.streamlit/
echo "\
[theme]\n\
base='dark'\n\
" > ~/.streamlit/config.toml

python -m spacy download en_core_web_md 