import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import io
import os
import tempfile
from script2 import (cargar_estaciones, seleccionar_grupos, interpolar_puntos,
                     obtener_elevaciones_paralelo, calcular_pendiente_suavizada,
                     graficar_html, exportar_pdf, exportar_csv, exportar_kml, exportar_geojson)

st.set_page_config(page_title="Perfil Altimétrico Ferroviario", layout="wide")
st.title("🚆 Generador de Perfil Altimétrico Ferroviario - LAL 2025")

st.markdown("""
Subí un archivo **CSV** con estaciones o un **KML** con puntos (nombre, latitud, longitud).
El nombre del punto debe tener este formato: `NombreEstacion,KM` (ej: `Ayacucho,332.5`)
""")

# --- Función para procesar KML ---
def procesar_kml(kml_file):
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
                datos.append({
                    'Nombre': nombre.strip(),
                    'Km': float(km.strip()),
                    'Lat': float(lat),
                    'Lon': float(lon)
                })
            except Exception as e:
                st.warning(f"❌ Error en etiqueta '{name_tag.text}': {e}")

    if datos:
        return pd.DataFrame(datos)
    else:
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

# --- Mostrar tabla editable ---
if not df_estaciones.empty:
    st.subheader("📋 Estaciones cargadas")
    df_editada = st.data_editor(df_estaciones, use_container_width=True, num_rows="dynamic")

    if st.button("🚀 Generar perfil altimétrico"):
        try:
            with st.spinner("Procesando..."):
                # Guardar CSV temporal
                csv_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                df_editada.to_csv(csv_temp.name, index=False)

                # Cargar estaciones desde CSV validado
                estaciones = cargar_estaciones(csv_temp.name)

                # Tramo único
                tramos = [estaciones]
                nombre_tramo = f"{estaciones[0].nombre} a {estaciones[-1].nombre}"
                puntos_interp = interpolar_puntos(estaciones)
                puntos_elev = obtener_elevaciones_paralelo(puntos_interp)

                kms = pd.Series([p.km for p in puntos_elev])
                elevs = pd.Series([p.elevation for p in puntos_elev])
                pendientes = calcular_pendiente_suavizada(kms.to_numpy(), elevs.to_numpy())

                fig = graficar_html(puntos_elev, estaciones, "perfil_temp.html",
                                    titulo=f"Perfil Altimétrico - {nombre_tramo}",
                                    slope_data=pendientes)

                st.success("✅ Perfil generado")
                st.plotly_chart(fig, use_container_width=True)

                # Exportaciones
                nombre_base = f"perfil_{estaciones[0].nombre}_{estaciones[-1].nombre}".replace(" ", "_")

                col1, col2, col3 = st.columns(3)
                with col1:
                    pdf_file = f"{nombre_base}.pdf"
                    exportar_pdf(fig, pdf_file)
                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Descargar PDF", f, file_name=pdf_file)

                with col2:
                    csv_file = f"{nombre_base}_datos.csv"
                    exportar_csv(puntos_elev, pendientes, csv_file)
                    with open(csv_file, "rb") as f:
                        st.download_button("📊 Descargar CSV", f, file_name=csv_file)

                with col3:
                    kml_file = f"{nombre_base}_estaciones.kml"
                    exportar_kml(puntos_elev, estaciones, kml_file)
                    with open(kml_file, "rb") as f:
                        st.download_button("🌍 Descargar KML", f, file_name=kml_file)

                geojson_file = f"{nombre_base}.geojson"
                exportar_geojson(puntos_elev, estaciones, geojson_file)
                with open(geojson_file, "rb") as f:
                    st.download_button("🗺️ Descargar GeoJSON", f, file_name=geojson_file)

        except Exception as e:
            st.error(f"❌ Error al generar el perfil: {e}")
