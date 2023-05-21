from pydantic import BaseModel
from typing import List

class Song(BaseModel):
    id: str
    artist: str
    title: str
    albumUrl: str
    
class Preferences(BaseModel):
    user_id: str
    music: List[Song]
    speed: int
    mood: int
    emotion: int

class UserKnowledgeBases(BaseModel):
    user_id: str
    speed: int
    mood: int
    emotion: int

class ExistingUser(BaseModel):
    user_id: str

class SearchQuery(BaseModel):
    name: str

class SongsList(BaseModel):
    tracks: List[Song]

class LikedSong(BaseModel):
    user_id: str
    track_id: str

class LikedSongJson(BaseModel):
    user_id: str
    song: Song