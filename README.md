## Вимоги

* Python 3.11+
* pip, venv
* Docker

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Дані

Завантажити архіви з даними:
(дані на [https://drive.google.com/drive/folders/1aQVXzr80tZ-Y8htFMQfqfTrbQGpmx2pu?usp=sharing](https://drive.google.com/drive/folders/1aQVXzr80tZ-Y8htFMQfqfTrbQGpmx2pu?usp=sharing))

```bash
mkdir -p data
# розпакуйте архіви і покладіть в ./data
```

## Побудова індексів Chroma

```bash
python3 chroma_encoding/chromadb_formation.py
python3 chroma_encoding/chroma_parfumes.py
python3 chroma_encoding/chromadb_faq_formation.py
```

## Ініціалізація БД

```bash
python3 db_init.py
```

## Запуск

```bash
chmod +x run.sh
./run.sh
```

## Очищення Docker (за потреби)

```bash
./docker_nuke.sh
```

> Примітка: скрипти формують вектори та індекси в `./chroma_data` і створюють локальну БД для сервісу. Якщо змінюєте дані — перегенеруйте індекси й перевстановіть БД.
