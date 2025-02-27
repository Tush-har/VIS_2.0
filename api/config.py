from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "user_login"
USERS_COLLECTION = "user_data"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[USERS_COLLECTION]