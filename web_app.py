import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tempfile
from script2 import (cargar_estaciones, interpolar_puntos, obtener_elevaciones_paralelo,
                    calcular_pendiente_suavizada, graficar_html, exportar_pdf,
                    exportar_csv, exportar_kml, exportar_geojson)

st.set_page_config(page_title="Perfil Altimétrico Ferroviario", layout="wide")
st.title("🚆 Generador de Perfil Altimétrico Ferroviario - LAL 2025")

st.markdown("""
Subí un archivo **CSV** con estaciones o un **KML** con puntos (nombre, latitud, longitud).
El nombre del punto debe tener este formato: `NombreEstacion,KM` (ej: `Ayacucho,332.5`)
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
                datos.append({
                    'Nombre': nombre.strip(),
                    'Km': float(km.strip()),
                    'Lat': float(lat),
                    'Lon': float(lon)
                })
            except Exception as e:
                st.warning(f"❌ Error en etiqueta '{name_tag.text}': {e}")
    return pd.DataFrame(datos) if datos else pd.DataFrame()

# --- Generar perfil ---
@st.cache_data
def generar_perfil(_df_editada, intervalo_m, ventana_suavizado):
    """Genera el perfil altimétrico para un tramo."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as csv_temp:
        _df_editada.to_csv(csv_temp.name, index=False)
        estaciones = cargar_estaciones(csv_temp.name)
    
    tramos = [estaciones]
    nombre_tramo = f"{estaciones[0].nombre} a {estaciones[-1].nombre}"
    puntos_interp = interpolar_puntos(estaciones, intervalo_m)
    puntos_elev = obtener_elevaciones_paralelo(puntos_interp)
    kms = np.array([p.km for p in puntos_elev])
    elevs = np.array([p.elevation for p in puntos_elev])
    pendientes = calcular_pendiente_suavizada(kms, elevs, window_length=ventana_suavizado)
    fig = graficar_html(puntos_elev, estaciones, f"perfil_temp_{nombre_tramo}.html",
                        titulo=f"Perfil Altimétrico - {nombre_tramo}",
                        slope_data=pendientes)
    nombre_base = f"perfil_{estaciones[0].nombre}_{estaciones[-1].nombre}".replace(" ", "_")
    return fig, puntos_elev, pendientes, estaciones, nombre_base, csv_temp.name

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

    df_editada = st.data_editor(df_estaciones, use_container_width=True, num_rows="dynamic")
    
    st.subheader("⚙️ Configuración")
    intervalo_m = st.slider("Intervalo de interpolación (metros)", 50, 500, 100, step=10)
    ventana_suavizado = st.number_input("Ventana de suavizado", 3, 15, 5, step=2)

    if st.button("🚀 Generar perfil altimétrico"):
        try:
            with st.spinner("Procesando..."):
                fig, puntos_elev, pendientes, estaciones, nombre_base, csv_temp_path = generar_perfil(
                    df_editada, intervalo_m, ventana_suavizado
                )
                archivos_temporales = [f"perfil_temp_{nombre_base}.html"]

                st.success("✅ Perfil generado")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📥 Descargar resultados"):
                    pdf_file = f"{nombre_base}.pdf"
                    exportar_pdf(fig, pdf_file)
                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Descargar PDF", f, file_name=pdf_file)
                    archivos_temporales.append(pdf_file)

                    csv_file = f"{nombre_base}_datos.csv"
                    exportar_csv(puntos_elev, pendientes, csv_file)
                    with open(csv_file, "rb") as f:
                        st.download_button("📊 Descargar CSV", f, file_name=csv_file)
                    archivos_temporales.append(csv_file)

                    kml_file = f"{nombre_base}_estaciones.kml"
                    exportar_kml(puntos_elev, estaciones, kml_file)
                    with open(kml_file, "rb") as f:
                        st.download_button("🌍 Descargar KML", f, file_name=kml_file)
                    archivos_temporales.append(kml_file)

                    geojson_file = f"{nombre_base}.geojson"
                    exportar_geojson(puntos_elev, estaciones, geojson_file)
                    with open(geojson_file, "rb") as f:
                        st.download_button("🗺️ Descargar GeoJSON", f, file_name=geojson_file)
                    archivos_temporales.append(geojson_file)

        except ValueError as e:
            st.error(f"❌ Error en los datos: {e}")
        except FileNotFoundError as e:
            st.error(f"❌ Archivo no encontrado: {e}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
        finally:
            for archivo in [csv_temp_path] + archivos_temporales:
                try:
                    if os.path.exists(archivo):
                        os.remove(archivo)
                except:
                    pass
