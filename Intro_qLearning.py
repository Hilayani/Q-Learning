import numpy as np

class NavegadorNucleos:
    def __init__(self):
       
        self.nombres_nucleos = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                                'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S']
        

        self.matriz_adyacencia = np.array([
            # A  B  C  D  E  F  G  H  I  J  K  L  M  N  Ñ  O  P  Q  R  S
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A → B, D
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B → C, F, H
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C → E
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # D → R
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # E → G, J
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # F → G
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # G → K
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # H → I
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], # I → K, P
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # J → K
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # K → M, N
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # L → N, Ñ
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # M → L
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # N → M
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Ñ → O
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # O → 0 (sin salidas)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # P → Q
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Q → 0 (sin salidas)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # R → S
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # S → 0 (sin salidas)
        ])
        self.nucleo_a_indice = {nombre: i for i, nombre in enumerate(self.nombres_nucleos)}
    
    def mostrar_mapa_texto(self):
        """Muestra una representacion en texto del mapa de nucleos"""
        print("\n" + "="*60)
        
        for i, origen in enumerate(self.nombres_nucleos):
            destinos = []
            for j, destino in enumerate(self.nombres_nucleos):
                if self.matriz_adyacencia[i][j] == 1:
                    destinos.append(destino)
            if destinos:
                print(f"  {origen} -> {', '.join(destinos)}")
        
        print("="*60)
    
    def obtener_recorrido_optimizado(self, inicio_txt, destino_txt):
        """Q-Learning para encontrar la ruta optima"""
        gamma = 0.8
        alpha = 0.9
        
        idx_inicio = self.nucleo_a_indice[inicio_txt]
        idx_destino = self.nucleo_a_indice[destino_txt]
        
        if idx_inicio == idx_destino:
            return [inicio_txt]
        
        # Matriz de recompensas
        R = np.where(self.matriz_adyacencia == 0, -100.0, 0.0)
        for i in range(len(self.nombres_nucleos)):
            if self.matriz_adyacencia[i, idx_destino] == 1:
                R[i, idx_destino] = 500
        
        Q = np.zeros_like(R)
        
        # Entrenamiento
        for _ in range(5000):
            estado_actual = np.random.randint(0, 20)
            acciones_posibles = np.where(R[estado_actual] >= 0)[0]
            
            if len(acciones_posibles) > 0:
                accion = np.random.choice(acciones_posibles)
                max_q_siguiente = np.max(Q[accion])
                Q[estado_actual, accion] = R[estado_actual, accion] + gamma * max_q_siguiente
        
        # Trazar el camino
        recorrido = [inicio_txt]
        estado_actual = idx_inicio
        visitados = {idx_inicio}
        
        for _ in range(20):
            if estado_actual == idx_destino:
                return recorrido
            
            opciones = Q[estado_actual]
            siguiente_paso = np.argmax(opciones)
            
            if opciones[siguiente_paso] <= 0 or siguiente_paso in visitados:
                posibles = np.where(opciones > 0)[0]
                encontrado = False
                for p in posibles:
                    if p not in visitados:
                        siguiente_paso = p
                        encontrado = True
                        break
                if not encontrado:
                    return None
            
            estado_actual = siguiente_paso
            recorrido.append(self.nombres_nucleos[estado_actual])
            visitados.add(estado_actual)
        
        return None
    
    def ejecutar(self):
        print("VIAJE DE NUCLEOS")
        
        
        while True:
            print("\n" + "-"*60)
            print("Nucleos disponibles: " + ", ".join(self.nombres_nucleos))
            print("-"*60)
            
            inicio = input("Letra inicial: ").upper().strip()
            destino = input("Letra destino: ").upper().strip()
            
            if inicio in self.nombres_nucleos and destino in self.nombres_nucleos:
                ruta = self.obtener_recorrido_optimizado(inicio, destino)
                
                if ruta:
                    print("\n" + "="*60)
                    print("ruta existente:")
                    print("  " + " -> ".join(ruta))
                    print("="*60)
                else:
                    print("\n" + "!"*60)
                    print("No existe un camino desde", inicio, "hasta", destino)
                    print("!"*60)
            else:
                print("\n" + "!"*60)
                print("ERROR: Alguna de las letras ingresadas no existe")
                print("!"*60)
            
            print("\n" + "-"*40)
            opcion = input("quieres consultar otra ruta? (s/n): ").lower().strip()
            if opcion != 's':
                print("\n" + "="*60)
                print("bye pues madafaker!")
                print("="*60)
                break

# Ejecutar la aplicacion
if __name__ == "__main__":
    navegador = NavegadorNucleos()
    navegador.ejecutar()