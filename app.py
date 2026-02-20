import numpy as np
import streamlit as st

class NavegadorNucleos:
    def __init__(self):

        self.nombres_nucleos = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S'
        ]

        self.matriz_adyacencia = np.array([
            [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ])

        self.nucleo_a_indice = {
            nombre: i for i, nombre in enumerate(self.nombres_nucleos)
        }

    def obtener_recorrido_optimizado(self, inicio_txt, destino_txt):

        gamma = 0.8
        alpha = 0.9
        episodios = 8000

        idx_inicio = self.nucleo_a_indice[inicio_txt]
        idx_destino = self.nucleo_a_indice[destino_txt]

        if idx_inicio == idx_destino:
            return [inicio_txt]

        R = np.where(self.matriz_adyacencia == 1, 0.0, -100.0)

        for i in range(len(self.nombres_nucleos)):
            if self.matriz_adyacencia[i, idx_destino] == 1:
                R[i, idx_destino] = 500

        R[idx_destino, idx_destino] = 500

        Q = np.zeros_like(R)

        for _ in range(episodios):
            estado = np.random.randint(0, len(self.nombres_nucleos))

            while estado != idx_destino:
                acciones = np.where(R[estado] >= 0)[0]

                if len(acciones) == 0:
                    break

                accion = np.random.choice(acciones)
                max_q = np.max(Q[accion])

                Q[estado, accion] = Q[estado, accion] + alpha * (
                    R[estado, accion] + gamma * max_q - Q[estado, accion]
                )

                estado = accion

        ruta = [inicio_txt]
        estado = idx_inicio
        visitados = set()

        while estado != idx_destino:
            visitados.add(estado)
            siguiente = np.argmax(Q[estado])

            if Q[estado, siguiente] <= 0 or siguiente in visitados:
                return None

            estado = siguiente
            ruta.append(self.nombres_nucleos[estado])

        return ruta


# -------- STREAMLIT UI --------

st.title("Navegador de Núcleos con Q-Learning")

st.subheader("🗺️ Mapa de los nodos")
st.image("Mapa_Nodos.png", caption="Diagrama del sistema", use_container_width=True)

navegador = NavegadorNucleos()

inicio = st.selectbox("Núcleo inicial", navegador.nombres_nucleos)
destino = st.selectbox("Núcleo destino", navegador.nombres_nucleos)

if st.button("Buscar ruta óptima"):
    ruta = navegador.obtener_recorrido_optimizado(inicio, destino)

    if ruta:
        st.success("Ruta encontrada:")
        st.write(" → ".join(ruta))
    else:

        st.error("No existe un camino entre esos núcleos.")

