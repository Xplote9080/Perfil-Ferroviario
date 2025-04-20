# web_app.py
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import io
import os
import tempfile
from script2 import (cargar_estaciones, seleccionar_grupos, interpolar_puntos,
                    obtener_elevaciones_paralelo, calcular_pendiente_suavizada,
                    graficar_html, exportar_pdf, exportar_csv, exportar_kml, exportar_geojson)

# Configuración de la página
st.set_page_config(page_title="Perfil Altimétrico Ferroviario", layout="wide")
st.title("🚆 Generador de Perfil Altimétrico Ferroviario - LAL 2025")

st.markdown("""
Subí un archivo **CSV** con estaciones o un **KML** con puntos (nombre, latitud, longitud).
El nombre del punto debe tener este formato: `NombreEstacion,KM` (ej: `Ayacucho,332.5`)
""")

# --- Función para procesar KML ---
def procesar_kml(kml_file):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    try:
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
                    datos.append({
                        'Nombre': nombre.strip(),
                        'Km': float(km.strip()),
                        'Lat': float(lat),
                        'Lon': float(lon)
                    })
                except Exception as e:
                    st.warning(f"❌ Error en etiqueta '{name_tag.text}': {e}")

        return pd.DataFrame(datos) if datos else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error procesando KML: {e}")
        return pd.DataFrame()

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
                st.error("❌ El CSV debe tener las
