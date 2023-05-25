from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
from models import *

import spotipy
from dotenv import load_dotenv
from supabase import create_client, Client
from spotipy.oauth2 import SpotifyClientCredentials



def convert_df_to_songs(df: pd.DataFrame):
    res = []
    i = 0
    for _, row in df.iterrows():
        if(i < 50):
            song = Song(id=row['id'], artist=row['artist_name'], title=row['track_name'], albumUrl="")
            res.append(song)
        i += 1
    return res

def create_track(track: Song):
    track_exists = supabase.table("tracks").select("*").eq("id", track.id).execute()
    if len(track_exists.data) <= 0:
        sg = {"id": track.id, "artist": track.artist, "title": track.title, "albumUrl": track.albumUrl}
        supabase.table('tracks').insert(sg).execute()

def like_track(user_id: str, track: Song):
    user_already_liked = supabase.table("user_liked_tracks").select("*").eq("user_id", user_id).eq("track_id", track.id).execute()
    if len(user_already_liked.data) <= 0:
        lg = {"user_id": user_id, "track_id": track.id}
        supabase.table('user_liked_tracks').insert(lg).execute()

def change_user_preferences(user_prefs: UserKnowledgeBases):
    existing_preferences = supabase.table("user_preferences").select("*").eq("id", user_prefs.user_id).execute()
    if len(existing_preferences.data) <= 0:
        curr_prefs = {'id': user_prefs.user_id, 'mood': user_prefs.mood, 'speed': user_prefs.speed,'emotion': user_prefs.emotion,}
        supabase.table('user_preferences').insert(curr_prefs).execute()
    else:
        update_prefs = {'mood': user_prefs.mood, 'speed': user_prefs.speed,'emotion': user_prefs.emotion,}
        supabase.table("user_preferences").update(update_prefs).eq("id", user_prefs.user_id).execute()

def get_users_liked_songs(user_id: str):
    user_already_liked = supabase.table("user_liked_tracks").select("*").eq("user_id", user_id).execute()
    if len(user_already_liked.data) > 0:
        liked_songs = []
        for user_track in user_already_liked.data:
            track_id = user_track["track_id"]
            song_data = supabase.table("tracks").select("*").eq("id", track_id).execute()
            liked_songs.append(song_data.data[0])

        return liked_songs
    return None

def get_users_knowledge_bases(user_id: str):
    user_already_liked = supabase.table("user_preferences").select("*").eq("id", user_id).execute()
    if len(user_already_liked.data) > 0:
        return user_already_liked.data[0]
    return None


# https://jamspotapi-1-s6116691.deta.app
load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

td = pd.read_csv('tracks.csv')
tracksDf = pd.DataFrame(td)

# Set up the clients
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = Client(supabase_url, supabase_key)

spotify_client_id = os.getenv('SPOTIFY_SECRET')
spotify_secret = os.getenv('SPOTIFY_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


@app.get('/')
async def root():
    return {"hi": "Some msg", "data": 0}    
    
@app.post('/tracks')
async def get_cold_start_recommendations(sq: SearchQuery):
    mask = np.column_stack([tracksDf['track_name'].str.contains(sq.name, na=False, case=False) for col in tracksDf])
    df = tracksDf.loc[mask.any(axis=1)]
    found_tracks =  convert_df_to_songs(df)
    return SongsList(tracks=found_tracks)
    

@app.post('/recommend')
async def get_cold_start_recommendations(prefs: Preferences):
    with open('content_recommender.m5', 'rb') as f:
        content_recommender = pickle.load(f)
        
        song_titles = [track.title for track in prefs.music]

        content_recommender.set_music(song_titles)
        content_recommender.set_speed(prefs.speed)
        content_recommender.set_mood(prefs.mood)
        content_recommender.set_emotion(prefs.emotion)

        recommended_features = content_recommender.recommend_features()
        recommended_genres = content_recommender.recommend_genre()
        intersection = content_recommender.feature_genre_intersection(recommended_features, recommended_genres)
        result = content_recommender.get_total_score()

        user_prefs = UserKnowledgeBases(user_id=prefs.user_id, mood=prefs.mood, speed=prefs.speed, emotion=prefs.emotion)
        change_user_preferences(user_prefs)

        for track in prefs.music:
            sp_tr = sp.track(track.id)
            album_url = sp_tr['album']['images'][0]['url']
            song = Song(id=track.id, artist=track.artist, title=track.title, albumUrl=album_url)
            create_track(song)
            like_track(prefs.user_id, song)
            

        topN = []
        i = 0
        for _, row in result.iterrows():
            if(i < 30):
                track = sp.track(row['id'])
                track_cover = track['album']['images'][0]['url']

                song = Song(id=row['id'], artist=row['artist_name'], title=row['track_name'], albumUrl=track_cover)
                topN.append(song)
            i += 1

        return SongsList(tracks=topN)


@app.get('/recommend')
async def get_existing_user_recommendations(user_id: str = ""):
    likedSongs = get_users_liked_songs(user_id)
    bases = get_users_knowledge_bases(user_id)
    with open('content_recommender.m5', 'rb') as f:
        content_recommender = pickle.load(f)
        
        song_titles = [track["title"] for track in likedSongs]

        content_recommender.set_music(song_titles)
        content_recommender.set_speed(bases["speed"])
        content_recommender.set_mood(bases["mood"])
        content_recommender.set_emotion(bases["emotion"])

        recommended_features = content_recommender.recommend_features()
        recommended_genres = content_recommender.recommend_genre()
        intersection = content_recommender.feature_genre_intersection(recommended_features, recommended_genres)
        result = content_recommender.get_total_score()
            
        topN = []
        i = 0
        for _, row in result.iterrows():
            if(i < 30):
                track = sp.track(row['id'])
                track_cover = track['album']['images'][0]['url']

                song = Song(id=row['id'], artist=row['artist_name'], title=row['track_name'], albumUrl=track_cover)
                topN.append(song)
            i += 1

        return SongsList(tracks=topN)

@app.get('/liked_tracks')
async def get_current_liked_songs(user_id: str = ""):
    likedSongs = get_users_liked_songs(user_id)
    songsList = [Song(id=track["id"], artist=track["artist"], title=track["title"], albumUrl=track["albumUrl"]) for track in likedSongs]

    return SongsList(tracks=songsList)

@app.get('/recommend-explore')
async def get_diversified_recommendations(user_id: str = ""):
    likedSongs = get_users_liked_songs(user_id)
    bases = get_users_knowledge_bases(user_id)

    result_content = pd.DataFrame()
    with open('content_recommender.m5', 'rb') as f:
        content_recommender = pickle.load(f)
        
        song_titles = [track["title"] for track in likedSongs]

        content_recommender.set_music(song_titles)
        content_recommender.set_speed(bases["speed"])
        content_recommender.set_mood(bases["mood"])
        content_recommender.set_emotion(bases["emotion"])

        recommended_features = content_recommender.recommend_features()
        recommended_genres = content_recommender.recommend_genre()
        intersection = content_recommender.feature_genre_intersection(recommended_features, recommended_genres)
        result_content = content_recommender.get_total_score()
            
        
    with open('kmean_recommender.m5', 'rb') as f:
        kmean = pickle.load(f)
        kmean.scale_data_with_user(result_content.head(20))
        kmean.create_clustering_model()
        result = kmean.predict_users_playlist()

        rec_songs = []
        for _, row in result.iterrows():
            track = sp.track(row['id'])
            track_cover = track['album']['images'][0]['url']

            song = Song(id=row['id'], artist=row['artist_name'], title=row['track_name'], albumUrl=track_cover)
            rec_songs.append(song)

        return SongsList(tracks=rec_songs)
    
@app.post('/songs')
async def user_modify_like(liked_song_json: LikedSongJson):
    user_id = liked_song_json.user_id
    track_id = liked_song_json.song.id
    create_track(liked_song_json.song)

    user_already_liked = supabase.table("user_liked_tracks").select("*").eq("user_id", user_id).eq("track_id", track_id).execute()
    if len(user_already_liked.data) <= 0:
        lg = {"user_id": user_id, "track_id": track_id}
        supabase.table('user_liked_tracks').insert(lg).execute()
    else:
        supabase.table("user_liked_tracks").delete().eq("user_id", user_id).eq("track_id", track_id).execute()
        
