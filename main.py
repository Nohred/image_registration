import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Aplicar transformación afín (rotación, escala, traslación) ---
def apply_affine_transform(img, theta, scale, tx, ty):
    rows, cols = img.shape
    # Matriz de rotación + escala centrada
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.degrees(theta), scale)
    # Añadir traslación
    M[0, 2] += tx
    M[1, 2] += ty
    # Aplicar la transformación
    transformed = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed

# --- Función objetivo: error cuadrático medio ---
def objective_function(params, img_ref, img_mov):
    theta, scale, tx, ty = params
    transformed = apply_affine_transform(img_mov, theta, scale, tx, ty)
    mse = np.mean((img_ref - transformed) ** 2)
    penalty = 0.001 * abs(theta)  # penalización por rotar demasiado
    return mse + penalty

# --- Algoritmo EMDA básico ---
def EMDA(img_ref, img_mov, pop_size=30, generations=100):
    bounds = [(-np.pi, np.pi), (0.5, 2.0), (-50, 50), (-50, 50)]  # theta, scale, tx, ty
    dim = len(bounds)

    # Inicializar población
    population = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(pop_size)
    ])
    fitness = np.array([objective_function(ind, img_ref, img_mov) for ind in population])

    for gen in range(generations):
        for i in range(pop_size):
            # Seleccionar 3 individuos distintos
            idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = population[idxs]
            F = 0.8  # factor de escala
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])

            # Crossover binomial
            CR = 0.9
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

            # Evaluar y seleccionar
            trial_fitness = objective_function(trial, img_ref, img_mov)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

    # Mejor solución
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# --- Función principal ---
def main():
    # Cargar imágenes PGM en escala de grises
    img_ref = cv2.imread('./data/I_1.png', cv2.IMREAD_GRAYSCALE)
    img_mov = cv2.imread('./data/I_4.png', cv2.IMREAD_GRAYSCALE)

    if img_ref is None or img_mov is None:
        raise ValueError("Error al cargar imágenes")

    # Normalizar a [0, 1]
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    # Ejecutar EMDA
    print("Ejecutando EMDA...")
    best_params, best_score = EMDA(img_ref, img_mov)
    print("\nMejores parámetros encontrados:")
    print(f"Ángulo (radianes): {best_params[0]:.4f}")
    print(f"Escala: {best_params[1]:.4f}")
    print(f"Traslación x: {best_params[2]:.2f} px")
    print(f"Traslación y: {best_params[3]:.2f} px")
    print(f"Error cuadrático medio: {best_score:.6f}")

    # Transformar imagen con mejores parámetros
    result = apply_affine_transform(img_mov, *best_params)

    # Mostrar resultados
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Referencia")
    plt.imshow(img_ref, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Movida")
    plt.imshow(img_mov, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Registrada")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Ejecutar si es principal ---
if __name__ == "__main__":
    main()
