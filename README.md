python3 -m venv .venv
pip install -r requirements.txt

source .venv/bin/activate
python3 chroma_encoding/chromadb_formation.py 
python3 chroma_encoding/chroma_parfumes.py 
python3 chroma_encoding/chromadb_faq_formation.py 

mkdir data

python3 db_init.py

./run.sh (chmod +x)

./docker_nuke.sh if needed
