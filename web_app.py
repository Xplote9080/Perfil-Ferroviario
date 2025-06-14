import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tempfile
import numpy as np
from script2 import (cargar_estaciones, interpolar_puntos,
                    obtener_elevaciones_paralelo, calcular_pendiente_suavizada,
                    graficar_html, exportar_pdf, exportar_csv, exportar_kml, exportar_geojson)

# Configuración de la página
st.set_page_config(page_title="Perfil Altimétrico Ferroviario", layout="wide")
st.title("🚆 Generador de Perfil Altimétrico Ferroviario - LAL 2025")

st.markdown("""
Subí un archivo **CSV** con estaciones o un **KML** con puntos (nombre, latitud, longitud).
El nombre del punto debe tener este formato: `NombreEstacion,KM` (ej: `Ayacucho,332.5`).
""")

# --- Función para procesar KML ---
def procesar_kml(kml_file):
    """Procesa un archivo KML para extraer estaciones."""
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file)
    root = tree.getroot()
    placemarks = root.findall('.//kml:Placemark', ns)
    datos = []
    for pm in placemarks:
        name_tag = pm.find('kml:name', ns)
        point_tag = pm.find('.//kml:Point', ns)
        coord_tag = point_tag.find('kml:coordinates', ns) if point_tag is not None else None
        if name_tag is not None and coord_tag is not None:
            try:
                texto = name_tag.text.strip()
                nombre, km = texto.split(',')
                lon, lat, *_ = coord_tag.text.strip().split(',')
                lat = float(lat)
                lon = float(lon)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError(f"Coordenadas fuera de rango: Lat={lat}, Lon={lon}")
                datos.append({
                    'Nombre': nombre.strip(),
                    'Km': float(km.strip()),
                    'Lat': lat,
                    'Lon': lon
                })
            except Exception as e:
                st.warning(f"❌ Error en etiqueta '{name_tag.text}': {e}")
    df = pd.DataFrame(datos)
    if not df.empty:
        # Validar orden de kilómetros
        kms = df['Km'].values
        if len(kms) < 2:
            st.error("❌ Se requieren al menos 2 estaciones")
            return pd.DataFrame()
        if not all(kms[i] < kms[i+1] for i in range(len(kms)-1)):
            st.error("❌ Los kilómetros deben ser estrictamente crecientes")
            return pd.DataFrame()
    return df

# --- Generar perfil ---
@st.cache_data
def generar_perfil(_df_editada, intervalo_m, ventana_suavizado):
    """Genera el perfil altimétrico para un tramo."""
    progress_bar = st.progress(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as csv_temp:
        _df_editada.to_csv(csv_temp.name, index=False)
        estaciones = cargar_estaciones(csv_temp.name)
    
    tramos = [estaciones]
    nombre_tramo = f"{estaciones[0].nombre} a {estaciones[-1].nombre}"
    puntos_interp = interpolar_puntos(estaciones, intervalo_m)
    puntos_elev = obtener_elevaciones_paralelo(puntos_interp, progress_callback=progress_bar.progress)
    kms = np.array([p.km for p in puntos_elev])
    elevs = np.array([p.elevation for p in puntos_elev])
    pendientes = calcular_pendiente_suavizada(kms, elevs, window_length=ventana_suavizado)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        html_file = tmp.name
        fig = graficar_html(puntos_elev, estaciones, html_file,
                            titulo=f"Perfil Altimétrico - {nombre_tramo}",
                            slope_data=pendientes)
    nombre_base = f"perfil_{estaciones[0].nombre}_{estaciones[-1].nombre}".replace(" ", "_")
    return fig, puntos_elev, pendientes, estaciones, nombre_base, csv_temp.name, html_file

# --- Carga del archivo ---
archivo_subido = st.file_uploader("📤 Subí tu archivo CSV o KML", type=['csv', 'kml'])
df_estaciones = pd.DataFrame()

if archivo_subido:
    nombre_archivo = archivo_subido.name
    if nombre_archivo.lower().endswith('.csv'):
        try:
            df_estaciones = pd.read_csv(archivo_subido)
            df_estaciones.columns = [c.strip().capitalize() for c in df_estaciones.columns]
            if set(['Nombre', 'Km', 'Lat', 'Lon']).issubset(df_estaciones.columns):
                st.success("✅ CSV válido cargado")
            else:
                st.error("❌ El CSV debe tener las columnas: Nombre, Km, Lat, Lon")
                df_estaciones = pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error leyendo el CSV: {e}")
    elif nombre_archivo.lower().endswith('.kml'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            tmp.write(archivo_subido.read())
            tmp_path = tmp.name
        df_estaciones = procesar_kml(tmp_path)
        if not df_estaciones.empty:
            st.success(f"✅ {len(df_estaciones)} estaciones extraídas del KML")
        else:
            st.error("❌ No se encontraron datos válidos en el KML")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Configuración y procesamiento ---
if not df_estaciones.empty:
    st.subheader("📋 Estaciones cargadas")
    with st.expander("🔍 Previsualizar datos"):
        st.dataframe(df_estaciones)
        try:
            validar_datos = df_estaciones.copy()
            validar_datos["Km"] = pd.to_numeric(validar_datos["Km"], errors="raise")
            validar_datos["Lat"] = pd.to_numeric(validar_datos["Lat"], errors="raise")
            validar_datos["Lon"] = pd.to_numeric(validar_datos["Lon"], errors="raise")
            if (validar_datos["Lat"].between(-90, 90)).all() and (validar_datos["Lon"].between(-180, 180)).all():
                st.success("✅ Datos válidos")
            else:
                st.error("❌ Algunas latitudes o longitudes están fuera de rango")
        except ValueError:
            st.error("❌ Error: Algunos valores de Km, Lat o Lon no son numéricos")

    # --- Selección de estaciones ---
    st.subheader("📍 Seleccionar Estaciones")
    estaciones_disponibles = df_estaciones['Nombre'].tolist()
    estaciones_seleccionadas = st.multiselect(
        "Selecciona 2 o más estaciones para el perfil altimétrico",
        options=estaciones_disponibles,
        default=estaciones_disponibles[:2],  # Selecciona las primeras 2 por defecto
        help="Elige 2 o más estaciones. Las estaciones deben estar en orden creciente de Km."
    )

    if len(estaciones_seleccionadas) < 2:
        st.warning("⚠️ Por favor, selecciona al menos 2 estaciones.")
    else:
        # Filtrar el DataFrame para incluir solo las estaciones seleccionadas
        df_filtrado = df_estaciones[df_estaciones['Nombre'].isin(estaciones_seleccionadas)].copy()
        # Ordenar por Km para asegurar orden creciente
        df_filtrado = df_filtrado.sort_values(by='Km')
        # Verificar que las estaciones seleccionadas estén en orden creciente de Km
        kms = df_filtrado['Km'].values
        if not all(kms[i] < kms[i+1] for i in range(len(kms)-1)):
            st.error("❌ Las estaciones seleccionadas deben estar en orden creciente de Km.")
        else:
            st.success(f"✅ Estaciones seleccionadas: {', '.join(estaciones_seleccionadas)}")
            df_editada = st.data_editor(df_filtrado, use_container_width=True, num_rows="dynamic")

            st.subheader("⚙️ Configuración")
            intervalo_m = st.slider("Intervalo de interpolación (metros)", 50, 500, 200, step=10)
            ventana_suavizado = st.number_input("Ventana de suavizado", 3, 15, 5, step=2)

            # Botón para limpiar el caché
            with st.expander("🛠️ Opciones avanzadas"):
                if st.button("🗑️ Limpiar caché de elevaciones"):
                    if os.path.exists("elevations_cache.csv"):
                        os.remove("elevations_cache.csv")
                        st.success("✅ Caché eliminado")
                    else:
                        st.info("ℹ️ No hay caché para eliminar")
                if os.path.exists("elevations_cache.csv"):
                    with open("elevations_cache.csv", "rb") as f:
                        st.download_button("📥 Descargar caché de elevaciones", f, file_name="elevations_cache.csv")

            if st.button("🚀 Generar perfil altimétrico"):
                try:
                    with st.spinner("Procesando..."):
                        fig, puntos_elev, pendientes, estaciones, nombre_base, csv_temp_path, html_file = generar_perfil(
                            df_editada, intervalo_m, ventana_suavizado
                        )
                        archivos_temporales = [csv_temp_path, html_file]

                        st.success("✅ Perfil generado")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("""
                        **Instrucciones para ajustar el gráfico:**
                        - **Zoom en el gráfico**: Usa la rueda del ratón o selecciona un área para acercar.
                        - **Desplazamiento**: Haz clic y arrastra el gráfico para moverlo.
                        - **Ajustar el eje Y (elevación)**:
                          - **Desplazar**: Haz clic y arrastra el eje Y para moverlo hacia arriba o abajo.
                          - **Escalar**: Usa las herramientas de zoom en la barra superior o selecciona un rango en el eje Y.
                          - **Restablecer**: Haz clic en 'Reset axes' en la barra superior para volver al rango original.
                        """)

                        with st.expander("📥 Descargar resultados"):
                            # Descarga HTML
                            if os.path.exists(html_file):
                                with open(html_file, "rb") as f:
                                    st.download_button("🌐 Descargar HTML", f, file_name=f"{nombre_base}.html")
                            else:
                                st.error("❌ No se pudo encontrar el archivo HTML para descargar.")

                            # Descarga PDF
                            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                pdf_file = tmp.name
                                exportar_pdf(fig, pdf_file)
                                archivos_temporales.append(pdf_file)
                            with open(pdf_file, "rb") as f:
                                st.download_button("📄 Descargar PDF", f, file_name=f"{nombre_base}.pdf")

                            # Descarga CSV
                            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                                csv_file = tmp.name
                                exportar_csv(puntos_elev, pendientes, csv_file)
                                archivos_temporales.append(csv_file)
                            with open(csv_file, "rb") as f:
                                st.download_button("📊 Descargar CSV", f, file_name=f"{nombre_base}_datos.csv")

                            # Descarga KML
                            with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
                                kml_file = tmp.name
                                exportar_kml(puntos_elev, estaciones, kml_file)
                                archivos_temporales.append(kml_file)
                            with open(kml_file, "rb") as f:
                                st.download_button("🌍 Descargar KML", f, file_name=f"{nombre_base}_estaciones.kml")

                            # Descarga GeoJSON
                            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                                geojson_file = tmp.name
                                exportar_geojson(puntos_elev, estaciones, geojson_file)
                                archivos_temporales.append(geojson_file)
                            with open(geojson_file, "rb") as f:
                                st.download_button("🗺️ Descargar GeoJSON", f, file_name=f"{nombre_base}.geojson")

                except ValueError as e:
                    st.error(f"❌ Error en los datos: {e}")
                except FileNotFoundError as e:
                    st.error(f"❌ Archivo no encontrado: {e}")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {e}")
                finally:
                    for archivo in archivos_temporales:
                        try:
                            if os.path.exists(archivo):
                                os.remove(archivo)
                        except:
                            pass
