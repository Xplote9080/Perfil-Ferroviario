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
                st.error("❌ El CSV debe tener las columnas: Nombre, Km, Lat, Lon")
                df_estaciones = pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error leyendo el CSV: {e}")

    elif nombre_archivo.lower().endswith('.kml'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            tmp.write(archivo_subido.read())
            tmp_path = tmp.name
        df_estaciones = procesar_kml(tmp_path)
        try:
            os.unlink(tmp_path)  # Eliminar archivo temporal
        except:
            pass
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
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as csv_temp:
                    df_editada.to_csv(csv_temp.name, index=False)
                    csv_path = csv_temp.name

                # Cargar estaciones desde CSV validado
                estaciones = cargar_estaciones(csv_path)

                # Tramo único
                tramos = [estaciones]
                nombre_tramo = f"{estaciones[0].nombre} a {estaciones[-1].nombre}"
                puntos_interp = interpolar_puntos(estaciones)
                if not puntos_interp:
                    raise ValueError("No se pudieron interpolar puntos")

                puntos_elev = obtener_elevaciones_paralelo(puntos_interp, author="Perfil altimétrico ferroviario - LAL 2025")
                if not puntos_elev:
                    raise ValueError("No se pudieron obtener elevaciones")

                kms = pd.Series([p.km for p in puntos_elev])
                elevs = pd.Series([p.elevation if p.elevation is not None else 0.0 for p in puntos_elev])
                pendientes = calcular_pendiente_suavizada(kms.to_numpy(), elevs.to_numpy())

                # Generar gráfico
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as html_temp:
                    fig = graficar_html(
                        puntos_elev, estaciones, html_temp.name,
                        titulo=f"Perfil Altimétrico - {nombre_tramo}",
                        slope_data=pendientes,
                        theme="light", colors="blue,red", watermark="LAL"
                    )
                    html_path = html_temp.name

                if fig is None:
                    raise ValueError("No se pudo generar el gráfico")

                st.success("✅ Perfil generado")
                st.plotly_chart(fig, use_container_width=True)

                # Exportaciones
                nombre_base = f"perfil_{estaciones[0].nombre}_{estaciones[-1].nombre}".replace(" ", "_")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pdf_file = f"{nombre_base}.pdf"
                    exportar_pdf(fig, pdf_file)
                    if os.path.exists(pdf_file):
                        with open(pdf_file, "rb") as f:
                            st.download_button("📄 Descargar PDF", f, file_name=pdf_file)
                    else:
                        st.warning("No se pudo generar el PDF")

                with col2:
                    csv_file = f"{nombre_base}_datos.csv"
                    exportar_csv(puntos_elev, pendientes, csv_file, author="Perfil altimétrico ferroviario - LAL 2025")
                    if os.path.exists(csv_file):
                        with open(csv_file, "rb") as f:
                            st.download_button("📊 Descargar CSV", f, file_name=csv_file)
                    else:
                        st.warning("No se pudo generar el CSV")

                with col3:
                    kml_file = f"{nombre_base}_estaciones.kml"
                    exportar_kml(puntos_elev, estaciones, kml_file, author="Perfil altimétrico ferroviario - LAL 2025")
                    if os.path.exists(kml_file):
                        with open(kml_file, "rb") as f:
                            st.download_button("🌍 Descargar KML", f, file_name=kml_file)
                    else:
                        st.warning("No se pudo generar el KML")

                with col4:
                    geojson_file = f"{nombre_base}_datos.geojson"
                    exportar_geojson(puntos_elev, estaciones, geojson_file, author="Perfil altimétrico ferroviario - LAL 2025")
                    if os.path.exists(geojson_file):
                        with open(geojson_file, "rb") as f:
                            st.download_button("🗺️ Descargar GeoJSON", f, file_name=geojson_file)
                    else:
                        st.warning("No se pudo generar el GeoJSON")

                # Limpiar archivos temporales
                for temp_file in [csv_path, html_path, pdf_file, csv_file, kml_file, geojson_file]:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except:
                        pass

        except Exception as e:
            st.error(f"❌ Error al generar el perfil: {e}")

else:
    st.info("Por favor, subí un archivo CSV o KML para comenzar.")
