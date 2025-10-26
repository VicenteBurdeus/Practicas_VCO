import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def distance(p1, p2):
    """Calcular distancia euclidiana entre dos puntos"""
    return np.sqrt(np.sum((p1-p2)**2))

def assign_clusters(X, clusters, k):
    """Asignar puntos a clusters"""
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(clusters, k):
    """Actualizar centros de clusters"""
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters

def pred_cluster(X, clusters, k):
    """Predecir cluster para cada punto"""
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return np.array(pred)

def initialize_clusters(k, X):
    """Inicializar clusters aleatoriamente"""
    clusters = {}
    np.random.seed(23)
    
    for idx in range(k):
        center = 2*(2*np.random.random((X.shape[1],))-1)
        cluster = {
            'center': center,
            'points': []
        }
        clusters[idx] = cluster
    return clusters

def run_kmeans(X, k, max_iters=100):
    """Ejecutar algoritmo k-means"""
    # Inicializar clusters
    clusters = initialize_clusters(k, X)
    
    # Iterar hasta convergencia o número máximo de iteraciones
    for _ in range(max_iters):
        old_centers = np.array([clusters[i]['center'] for i in range(k)])
        
        # Asignar puntos a clusters
        clusters = assign_clusters(X, clusters, k)
        
        # Actualizar centros
        clusters = update_clusters(clusters, k)
        
        # Verificar convergencia
        new_centers = np.array([clusters[i]['center'] for i in range(k)])
        if np.allclose(old_centers, new_centers):
            break
    
    return clusters, pred_cluster(X, clusters, k)

# Generar datos
n_samples = 500
n_features = 3
random_state = 23
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, random_state=random_state)

# Crear figura para mostrar datos originales y resultados
plt.figure(figsize=(15, 10))

# Mostrar datos originales
plt.subplot(221)
plt.scatter(X[:,0], X[:,1])
plt.title('Datos Originales')
plt.grid(True)

# Ejecutar k-means con diferentes valores de k
k_values = [3, 4, 5]

for idx, k in enumerate(k_values, 1):
    # Ejecutar k-means
    clusters, pred = run_kmeans(X, k)
    
    # Mostrar resultados
    plt.subplot(2, 2, idx+1)
    plt.scatter(X[:,0], X[:,1], c=pred, cmap='viridis')
    
    # Mostrar centros
    for i in range(k):
        center = clusters[i]['center']
        plt.scatter(center[0], center[1], marker='^', c='red', s=200, label=f'Centro {i+1}')
    
    plt.title(f'K-means con k={k}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Mostrar información adicional
print("\nEvaluación de diferentes valores de k:")
for k in k_values:
    clusters, pred = run_kmeans(X, k)
    # Calcular la suma de las distancias al cuadrado dentro de cada cluster
    total_dist = 0
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if len(points) > 0:
            distances = np.sum((points - clusters[i]['center'])**2, axis=1)
            total_dist += np.sum(distances)
    print(f"k={k}: Suma total de distancias al cuadrado = {total_dist:.2f}")
