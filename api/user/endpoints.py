import os
from datetime import timedelta
from typing import Optional
from config import collection
from fastapi import HTTPException, status, APIRouter
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

@router.post("/add_user_account")
async def register(user: dict):
    existing_username = collection.find_one({"username": user['username']})
    existing_email = collection.find_one({"email": user['email']})
    if existing_username or existing_email:
        raise HTTPException(status_code=status.HTTP_208_ALREADY_REPORTED, detail="Username already registered")

    hashed_password = pwd_context.hash(user['password'])

    user_dict = {'username': user['username'], "email": user['email'], "password": hashed_password,
                  "role_name":user['role_name'], "active" : True}
    result = collection.insert_one(user_dict)
    user_id = str(result.inserted_id)

    return {"id": user_id, "username": user['username']}


@router.post("/login")
async def login(user: dict):
    db_user = collection.find_one({"email" : user['email']})

    if not db_user and db_user['active'] is True:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Username not found")

    if not pwd_context.verify(user['password'], db_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password")
    
    return {"response" : "Successful login"}

@router.put("/update_password")
async def update_password(user_data: dict):
    db_user = collection.find_one({"email" : user_data['email']})
    if not db_user and db_user['active'] is True:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Username not found")

    if not pwd_context.verify(user_data['current_password'], db_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

    hashed_password = pwd_context.hash(user_data['new_password'])
    
    update_result = collection.update_one(
        {"username": user_data['username']},
        {"$set": {"password": hashed_password}}
    )

    if update_result.modified_count == 1:
        return {"message": "Password updated successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update password")



@router.delete("/delete_password")
async def delete_password(user_data: dict):
    db_user = collection.find_one({"username": user_data['username']})
    if not db_user and db_user['active'] is True:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Username not found")

    if not pwd_context.verify(user_data['password'], db_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")
    
    update_result = collection.update_one(
        {"username": user_data['username']},
        {"$set": {"active": False}}
    )

    if update_result.modified_count == 1:
        return {"message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User not deleted successfully")
    

@router.get('/user_accounts')
async def get_user():
    try:
        users = []
        for document in collection.find({}):
            if 'username' in document:
                users.append({"username" : document['username'],"email" : document['email'], "role_name" : document['role_name']})
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")
        
