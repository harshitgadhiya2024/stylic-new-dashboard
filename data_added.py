data = 

from pymongo import MongoClient

client = MongoClient("mongodb+srv://infostylicai:gUgH6G9oDimFRWIS@stylicai.tio3ghn.mongodb.net/")
db = client["stylicai"]
collection = db["model_faces"]

data_new = []
data_new_keys = list(data.keys())
for background in data_new_keys:
    record = data[background]["record"]
    record["background_name"] = record["background_name"].replace("_", " ")
    data_new.append(record)

collection.insert_many(data_new)

