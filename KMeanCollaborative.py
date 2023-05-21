import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from IPython.display import display
from math import pi, ceil


def make_radar(row, title, color, dframe, num_clusters):
    # number of variable
    categories=list(dframe)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the radar plot
    #ax = plt.subplot(4,ceil(num_clusters/4),row+1, polar=True, )
    ax = plt.subplot(2,ceil(num_clusters/2),row+1, polar=True, )
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=14)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=8)
    plt.ylim(0,1)

    # Ind1
    values=dframe.loc[row].drop('cluster').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=16, color=color, y=1.06)




class KMeanCollaborative:

    def __init__(self, data):
        self.initial_data = data
        self.data = pd.DataFrame()
        self.user_playlist = pd.DataFrame()
        self.initial_scaled_data = None
        self.initial_user_scaled = None
        self.model = None
        self.no_clusters = 0
        self.columns_scaled = None

    def scale_data_with_user(self, user_playlist):
        self.user_playlist = user_playlist
        x_df = self.initial_data[['danceability', 'energy', 'acousticness', 'valence', 'tempo']].values 
        x_user = user_playlist[['danceability', 'energy','acousticness', 'valence', 'tempo']].values 
        min_max_scaler = MinMaxScaler()
        x_df_scaled = min_max_scaler.fit_transform(x_df)
        x_user_scaled = min_max_scaler.fit_transform(x_user)

        self.columns_scaled = ['danceability_scaled', 'energy_scaled', 'acousticness_scaled','valence_scaled', 'tempo_scaled']

        self.initial_scaled_data = x_df_scaled
        self.initial_user_scaled = x_user_scaled
        self.data = pd.DataFrame(x_df_scaled, columns=self.columns_scaled)
        # self.user_playlist = pd.DataFrame(x_user_scaled, columns=self.columns_scaled)

    
    def analyze_data(self):
        n_clusters = range(2,21)
        ssd = []
        sc = []

        for n in n_clusters:
            km = KMeans(n_clusters=n, max_iter=300, n_init=10, init='k-means++', random_state=42)
            km.fit(self.initial_scaled_data)
            preds = km.predict(self.initial_scaled_data) 
            centers = km.cluster_centers_ 
            ssd.append(km.inertia_) 
            score = silhouette_score(self.initial_scaled_data, preds, metric='euclidean')
            sc.append(score)
            print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))

        # Analyze the Silhouette scores
        plt.plot(n_clusters, sc, marker='.', markersize=12, color='red')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette score behavior over the number of clusters')
        plt.show()

        # Get the Within-Cluster sum of squares
        for n, s in zip(n_clusters, ssd):
            print('Number of Clusters = {}, Sum of Squared Distances = {}'.format(n, s))

        plt.plot(n_clusters, ssd, marker='.', markersize=12)
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow method for optimal K')
        plt.show()

    
    def create_clustering_model(self, k=6):
        self.no_clusters = k
        self.model = KMeans(n_clusters=k, random_state=42).fit(self.initial_scaled_data)
        pred = self.model.predict(self.initial_scaled_data)
        self.data['cluster'] = self.model.labels_
    
    def visualize_clusters(self):
        self.data['cluster'].value_counts().plot(kind='bar')
        plt.xlabel('Cluster')
        plt.ylabel('Amount of songs')
        plt.title('Amount of songs per cluster')
        plt.show()

        df_songs_joined = pd.concat([self.data, self.initial_data], axis=1).set_index('cluster')

        for cluster in range(self.no_clusters):
            display(df_songs_joined.loc[cluster, ['artist_name','track_name']].sample(frac=1).head(10))

        df_radar = self.data.groupby('cluster').mean().reset_index()

        plt.figure(figsize=(24,15))
        my_palette = plt.cm.get_cmap("brg", len(df_radar.index))
        # Create cluster name
        title_list = ['Speech-Energy', 'Slow-Acoustic', 'Moderate-Instrumental-Voice-Happy', 'Voice-Moderate', 
                    'Voice-Dancy', 'Moderate-Instrumental-Voice-Sad']

        # Loop to plot
        for row in range(0, len(df_radar.index)):
            make_radar(row=row, title=str(df_radar['cluster'][row]) + ' : ' + title_list[row], 
                    color=my_palette(row), dframe=df_radar, num_clusters=len(df_radar.index))


    def predict_users_playlist(self, show_reports = False):
        user_pred = self.model.predict(self.initial_user_scaled)
        print('User Playlist clusters: ', user_pred)

        user_cluster = pd.DataFrame(self.initial_user_scaled, columns=self.columns_scaled)
        user_cluster['cluster'] = user_pred

        user_play_r = self.user_playlist.reset_index(drop=True)
        df_user_songs_joined = pd.concat([user_cluster,user_play_r], axis=1).set_index('cluster')

        df_user_songs_joined.reset_index(inplace=True)
        cluster_pct = df_user_songs_joined.cluster.value_counts(normalize=True)*20

        if int(cluster_pct.round(0).sum()) < 20:
            cluster_pct[cluster_pct < 0.5] = cluster_pct[cluster_pct < 0.5] + 1.0

        df_user_songs_joined['cluster_pct'] = df_user_songs_joined['cluster'].apply(lambda c: cluster_pct[c])
        df_user_songs_joined.drop(columns=self.columns_scaled, inplace=True)

        df_songs_joined = pd.concat([self.data,self.initial_data], axis=1).set_index('cluster')
        df_songs_joined = df_songs_joined.reset_index(drop=False)
        playlist = pd.DataFrame()

        for ncluster, pct in cluster_pct.items():
            songs = df_songs_joined[df_songs_joined['cluster'] == ncluster].sample(n=int(round(pct, 0)))
            playlist = pd.concat([playlist,songs], ignore_index=True)
            if len(playlist) > 20 :
                flag = 20 - len(playlist)
                playlist = playlist[:flag]

        if show_reports:
            user_cluster['cluster'].value_counts().plot(kind='bar', color='green')
            plt.xlabel('Cluster')
            plt.ylabel('Amount of songs')
            plt.title('Amount of songs in the users clusters')
            plt.show()

            for cluster in user_cluster['cluster'].unique():
                display(df_user_songs_joined.loc[cluster, ['artist_name','track_name']].sample(frac=1))

            display(cluster_pct)

        return playlist[['id', 'artist_name', 'track_name', 'cluster']]
            

