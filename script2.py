# -*- coding: utf-8 -*-

"""
Script para generar perfiles altimétricos a partir de datos de estaciones para vías férreas.
© 2025 LAL - Todos los derechos reservados.

Funcionalidades:
- Carga estaciones desde un archivo CSV.
- Permite seleccionar tramos (interactivo o vía CLI).
- Interpola puntos usando CubicSpline.
- Obtiene elevaciones con la API de Open-Meteo (con caché en CSV).
- Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay.
- Calcula desnivel acumulado (ascenso/descenso total).
- Genera gráficos HTML interactivos con Plotly (elevación + pendiente) con marca de agua.
- Exporta a PDF (opcional, requiere 'kaleido'), KML, CSV y GeoJSON.
- Valida rangos de latitud, longitud y kilómetros.
- Pregunta por nombre base al exportar; usa nombre consecutivo si no se especifica.
- Pregunta por título del gráfico; usa "Perfil Altimétrico - inicio a fin" por defecto.
- Incluye atribución en todos los outputs: "Perfil altimétrico ferroviario - LAL 2025".

Dependencias:
pip install requests numpy scipy tqdm plotly simplekml kaleido geojson tenacity
"""

import csv
import requests
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.offline import plot
import simplekml
from geojson import Point, Feature, FeatureCollection, dump
import logging
import time
import re
import os
import glob
from typing import List, Tuple, Optional, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_fixed
import argparse

# --- Configuración y Constantes ---
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CACHE_FILE = "elevations_cache.csv"
DEFAULT_STATIONS_FILE = "estaciones.csv"
DEFAULT_INTERVAL_METERS = 100  # 100 metros para mayor detalle
REQUEST_TIMEOUT_SECONDS = 15
MAX_API_WORKERS = 2
DEFAULT_ELEVATION_ON_ERROR = 0.0
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_OUTPUT_PREFIX = "perfil"
AUTHOR_ATTRIBUTION = "Perfil altimétrico ferroviario - LAL 2025"

# Configuración del Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Estructuras de Datos ---
class Station(NamedTuple):
    nombre: str
    km: float
    lat: float
    lon: float

class InterpolatedPoint(NamedTuple):
    km: float
    lat: float
    lon: float
    elevation: Optional[float] = None

# --- Funciones auxiliares ---
def get_next_output_name(output_dir: str, prefix: str = DEFAULT_OUTPUT_PREFIX) -> str:
    """Genera un nombre consecutivo basado en archivos existentes."""
    pattern = os.path.join(output_dir, f"{prefix}_*.*")
    existing_files = glob.glob(pattern)
    numbers = []
    for f in existing_files:
        match = re.match(rf".*{prefix}_(\d+)\..*", os.path.basename(f))
        if match:
            numbers.append(int(match.group(1)))
    next_number = max(numbers, default=0) + 1
    return f"{prefix}_{next_number}"

def sanitize_filename(filename: str) -> str:
    """Sanitiza nombres de archivo."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

# --- Funciones para caché ---
def _load_elevation_from_cache(lat: float, lon: float) -> Optional[float]:
    """Carga elevación desde el archivo CSV de caché."""
    if not os.path.exists(ELEVATION_CACHE_FILE):
        return None
    try:
        with open(ELEVATION_CACHE_FILE, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Saltar encabezado
            for row in reader:
                if len(row) >= 3 and float(row[0]) == lat and float(row[1]) == lon:
                    return float(row[2])
    except Exception as e:
        logging.warning(f"Error al leer caché CSV: {e}")
    return None

def _save_elevation_to_cache(lat: float, lon: float, elevation: float, author: str = AUTHOR_ATTRIBUTION):
    """Guarda elevación en el archivo CSV de caché."""
    try:
        file_exists = os.path.exists(ELEVATION_CACHE_FILE)
        with open(ELEVATION_CACHE_FILE, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([f"# {author} - Caché de elevaciones para vías férreas"])
                writer.writerow(["latitude", "longitude", "elevation"])
            writer.writerow([lat, lon, elevation])
    except Exception as e:
        logging.warning(f"Error al guardar en caché CSV: {e}")

# --- Funciones principales ---
def cargar_estaciones(archivo: str = DEFAULT_STATIONS_FILE) -> List[Station]:
    """Carga estaciones desde CSV con validaciones."""
    logging.info(f"Cargando estaciones desde: {archivo}")
    estaciones = []
    try:
        with open(archivo, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            required_cols = {"Nombre", "Km", "Lat", "Lon"}
            if not required_cols.issubset(reader.fieldnames):
                missing_cols = required_cols - set(reader.fieldnames)
                raise ValueError(f"Faltan columnas: {', '.join(missing_cols)}")

            for i, row in enumerate(reader):
                try:
                    nombre = row["Nombre"].strip()
                    km_str = row["Km"].strip().replace(',', '.')
                    lat_str = row["Lat"].strip().replace(',', '.')
                    lon_str = row["Lon"].strip().replace(',', '.')

                    if not nombre:
                        logging.warning(f"Fila {i+2}: Nombre vacío, omitiendo.")
                        continue
                    km = float(km_str)
                    lat = float(lat_str)
                    lon = float(lon_str)

                    if not (-90 <= lat <= 90):
                        raise ValueError(f"Latitud {lat} fuera de rango [-90, 90]")
                    if not (-180 <= lon <= 180):
                        raise ValueError(f"Longitud {lon} fuera de rango [-180, 180]")
                    if km < 0:
                        raise ValueError(f"Km {km} no puede ser negativo")

                    estaciones.append(Station(nombre=nombre, km=km, lat=lat, lon=lon))
                except KeyError as e:
                    raise ValueError(f"Fila {i+2}: Falta columna '{e}'")
                except ValueError as e:
                    raise ValueError(f"Fila {i+2}: Error en datos ({e})") from e

    except FileNotFoundError:
        logging.error(f"No se encontró: {archivo}")
        raise
    except ValueError as e:
        logging.error(f"Error en CSV: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        raise

    if not estaciones:
        raise ValueError(f"No se cargaron estaciones desde {archivo}")

    estaciones.sort(key=lambda s: s.km)
    logging.info(f"Cargadas {len(estaciones)} estaciones")
    return estaciones

def interpolar_puntos(estaciones: List[Station], intervalo_m: int = DEFAULT_INTERVAL_METERS) -> List[InterpolatedPoint]:
    """Interpola puntos geográficos."""
    if len(estaciones) < 2:
        logging.warning("Mínimo 2 estaciones requeridas")
        return []

    kms = np.array([s.km for s in estaciones])
    lats = np.array([s.lat for s in estaciones])
    lons = np.array([s.lon for s in estaciones])

    diff_kms = np.diff(kms)
    if np.any(diff_kms <= 0):
        logging.warning("Kms no crecientes, filtrando...")
        indices_validos = np.unique(np.concatenate((
            [0],
            np.where(diff_kms > 0)[0] + 1,
            [len(kms) - 1]
        )))
        kms = kms[indices_validos]
        lats = lats[indices_validos]
        lons = lons[indices_validos]
        if len(kms) < 2:
            logging.error("No hay suficientes puntos válidos")
            return []

    try:
        spline_lat = CubicSpline(kms, lats)
        spline_lon = CubicSpline(kms, lons)
    except ValueError as e:
        logging.error(f"Error en splines: {e}")
        return []

    km_inicio = kms[0]
    km_fin = kms[-1]
    num_puntos = max(2, int(np.round((km_fin - km_inicio) * 1000 / intervalo_m)) + 1)
    kms_interp = np.linspace(km_inicio, km_fin, num_puntos)
    lats_interp = spline_lat(kms_interp)
    lons_interp = spline_lon(kms_interp)

    puntos = [InterpolatedPoint(km=float(km_i), lat=float(lat_i), lon=float(lon_i))
              for km_i, lat_i, lon_i in zip(kms_interp, lats_interp, lons_interp)]
    logging.info(f"Interpolados {len(puntos)} puntos")
    return puntos

@lru_cache(maxsize=10000)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _obtener_elevacion_single(lat: float, lon: float, author: str = AUTHOR_ATTRIBUTION) -> Optional[float]:
    """Obtiene elevación desde API o caché con reintentos."""
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    elevation = _load_elevation_from_cache(lat_r, lon_r)
    if elevation is not None:
        logging.debug(f"Caché hit para ({lat_r:.5f}, {lon_r:.5f}): {elevation}")
        return elevation

    params = {'latitude': lat_r, 'longitude': lon_r}
    try:
        response = requests.get(ELEVATION_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        elevation = data.get("elevation")
        if isinstance(elevation, list) and len(elevation) > 0:
            elevation = float(elevation[0])
            _save_elevation_to_cache(lat_r, lon_r, elevation, author=author)
            return elevation
        logging.warning(f"Respuesta inesperada de API para ({lat_r:.5f}, {lon_r:.5f}): {data}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error en API para ({lat_r:.5f}, {lon_r:.5f}): {e}")
        raise

def obtener_elevaciones_paralelo(puntos: List[InterpolatedPoint], author: str = AUTHOR_ATTRIBUTION) -> List[InterpolatedPoint]:
    """Obtiene elevaciones en paralelo."""
    if not puntos:
        return []

    logging.info(f"Consultando elevaciones para {len(puntos)} puntos")
    puntos_actualizados = [None] * len(puntos)
    futures = {}
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        for i, punto in enumerate(puntos):
            future = executor.submit(_obtener_elevacion_single, punto.lat, punto.lon, author)
            futures[future] = i

        num_fallos = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Consultando elevaciones"):
            idx = futures[future]
            punto = puntos[idx]
            try:
                elevation = future.result()
                if elevation is None:
                    num_fallos += 1
                    elevation = DEFAULT_ELEVATION_ON_ERROR
                puntos_actualizados[idx] = punto._replace(elevation=elevation)
            except Exception as e:
                logging.error(f"Error en punto {idx}: {e}")
                num_fallos += 1
                puntos_actualizados[idx] = punto._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

    end_time = time.monotonic()
    logging.info(f"Elevaciones obtenidas en {end_time - start_time:.2f}s")
    if num_fallos > 0:
        logging.warning(f"Fallos en {num_fallos} puntos")
    return puntos_actualizados

def calcular_pendiente_suavizada(kms: np.ndarray, elevs: np.ndarray, window_length: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Calcula pendientes suavizadas con Savitzky-Golay."""
    slope = np.full_like(elevs, np.nan)
    valid_indices = ~np.isnan(elevs)
    if np.count_nonzero(valid_indices) < 2:
        logging.warning("No hay suficientes datos para calcular pendiente")
        return slope

    valid_kms = kms[valid_indices]
    valid_elevs = elevs[valid_indices]
    try:
        window = min(window_length, len(valid_elevs))
        if window % 2 == 0:
            window += 1
        if window < 3:
            window = 3
        elevs_smooth = savgol_filter(valid_elevs, window, polyorder=2)
        gradient_values = np.gradient(elevs_smooth, valid_kms)
        slope[valid_indices] = gradient_values
    except Exception as e:
        logging.error(f"Error al suavizar pendiente: {e}")
    return slope

def graficar_html(puntos_con_elevacion: List[InterpolatedPoint],
                  estaciones_tramo: List[Station],
                  archivo_html: str,
                  titulo: str = "Perfil altimétrico",
                  slope_data: Optional[np.ndarray] = None,
                  theme: str = "light",
                  colors: str = "blue,orange",
                  watermark: str = "LAL") -> Optional[go.Figure]:
    """Genera gráfico interactivo con tema, colores y marca de agua diagonal."""
    if not puntos_con_elevacion:
        logging.warning("No hay puntos para graficar")
        return None

    kms = np.array([p.km for p in puntos_con_elevacion])
    elevs = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion])
    
    try:
        elev_color, slope_color = colors.split(',')
    except ValueError:
        logging.warning("Formato de colores inválido, usando predeterminados")
        elev_color, slope_color = "blue", "orange"

    hover_texts = []
    for i, p in enumerate(puntos_con_elevacion):
        elev_text = f"{p.elevation:.1f}" if p.elevation is not None else "N/A"
        slope_text = f"{slope_data[i]:.1f}" if slope_data is not None and i < len(slope_data) and not np.isnan(slope_data[i]) else "N/A"
        hover_texts.append(f"<b>Km: {p.km:.3f}</b><br>Elev: {elev_text} m<br>Pendiente: {slope_text} m/km")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kms, y=elevs, mode='lines', name='Elevación',
        line=dict(color=elev_color, width=2),
        hoverinfo='text', text=hover_texts, yaxis='y1'
    ))

    kms_puntos = np.array([p.km for p in puntos_con_elevacion])
    elevs_puntos = np.array([p.elevation for p in puntos_con_elevacion])
    for est in estaciones_tramo:
        try:
            idx = np.nanargmin(np.abs(kms_puntos - est.km))
            elev_display = elevs_puntos[idx] if elevs_puntos[idx] is not None and not np.isnan(elevs_puntos[idx]) else 0.0
            slope_text = f"{slope_data[idx]:.1f}" if slope_data is not None and idx < len(slope_data) and not np.isnan(slope_data[idx]) else "N/A"
            fig.add_trace(go.Scatter(
                x=[kms_puntos[idx]], y=[elev_display],
                mode='markers+text', text=[est.nombre],
                textposition="top center",
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name=est.nombre,
                hoverinfo='text',
                hovertext=f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Elev: {elev_display:.1f} m<br>Pendiente: {slope_text} m/km",
                yaxis='y1'
            ))
        except (ValueError, IndexError):
            logging.warning(f"No se pudo añadir marcador para {est.nombre}")

    if slope_data is not None:
        fig.add_trace(go.Scatter(
            x=kms, y=slope_data, mode='lines', name='Pendiente (m/km)',
            line=dict(color=slope_color, width=1.5, dash='dash'),
            yaxis='y2', hoverinfo='skip'
        ))

    template = "plotly" if theme == "light" else "plotly_dark"
    annotations = []
    if watermark and watermark.lower() != "none":
        annotations.append(dict(
            text=watermark,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=60, color="rgba(150, 150, 150, 0.3)"),
            textangle=-45,
            align="center",
            opacity=0.3
        ))

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center'),
        xaxis_title="Kilómetro",
        yaxis=dict(
            title=dict(text="Elevación (msnm)", font=dict(color=elev_color)),
            tickfont=dict(color=elev_color)
        ),
        yaxis2=dict(
            title=dict(text="Pendiente (m/km)", font=dict(color=slope_color)),
            tickfont=dict(color=slope_color),
            anchor="x", overlaying="y", side="right", showgrid=False
        ),
        xaxis=dict(hoverformat='.3f'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=40),
        template=template,
        annotations=annotations
    )

    try:
        plot(fig, filename=archivo_html, auto_open=False)
        logging.info(f"Gráfico guardado en: {archivo_html}")
    except Exception as e:
        logging.error(f"Error al guardar HTML: {e}")
        return None
    return fig

def exportar_kml(puntos_con_elevacion: List[InterpolatedPoint],
                 estaciones_tramo: List[Station],
                 archivo_kml: str,
                 author: str = AUTHOR_ATTRIBUTION):
    """Exporta estaciones a KML con atribución."""
    if not estaciones_tramo or not puntos_con_elevacion:
        logging.warning("No hay datos para KML")
        return

    kml = simplekml.Kml(name=f"Estaciones: {estaciones_tramo[0].nombre} a {estaciones_tramo[-1].nombre}")
    kml.document.description = f"{author} - Perfil altimétrico para vías férreas"
    folder = kml.newfolder(name="Estaciones")
    kms_puntos = np.array([p.km for p in puntos_con_elevacion])
    elevs_puntos = np.array([p.elevation for p in puntos_con_elevacion])
    num_exportados = 0

    for est in estaciones_tramo:
        try:
            idx = np.nanargmin(np.abs(kms_puntos - est.km))
            elev_display = elevs_puntos[idx] if elevs_puntos[idx] is not None and not np.isnan(elevs_puntos[idx]) else 0.0
            pnt = folder.newpoint(name=est.nombre)
            pnt.coords = [(est.lon, est.lat, elev_display)]
            pnt.description = f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Elev: {elev_display:.1f} m<br>{author}"
            pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-stars.png'
            num_exportados += 1
        except Exception as e:
            logging.warning(f"No se pudo exportar {est.nombre}: {e}")

    if num_exportados > 0:
        try:
            kml.save(archivo_kml)
            logging.info(f"KML guardado en: {archivo_kml}")
        except Exception as e:
            logging.error(f"Error al guardar KML: {e}")

def exportar_geojson(puntos_con_elevacion: List[InterpolatedPoint],
                     estaciones_tramo: List[Station],
                     archivo_geojson: str,
                     author: str = AUTHOR_ATTRIBUTION):
    """Exporta puntos y estaciones a GeoJSON con atribución."""
    if not puntos_con_elevacion or not estaciones_tramo:
        logging.warning("No hay datos para GeoJSON")
        return

    features = []
    # Agregar puntos interpolados
    for p in puntos_con_elevacion:
        elevation = p.elevation if p.elevation is not None else None
        point = Point((p.lon, p.lat))
        properties = {
            "km": p.km,
            "elevation": elevation,
            "type": "interpolated"
        }
        features.append(Feature(geometry=point, properties=properties))

    # Agregar estaciones
    kms_puntos = np.array([p.km for p in puntos_con_elevacion])
    elevs_puntos = np.array([p.elevation for p in puntos_con_elevacion])
    for est in estaciones_tramo:
        try:
            idx = np.nanargmin(np.abs(kms_puntos - est.km))
            elev_display = elevs_puntos[idx] if elevs_puntos[idx] is not None and not np.isnan(elevs_puntos[idx]) else None
            point = Point((est.lon, est.lat))
            properties = {
                "name": est.nombre,
                "km": est.km,
                "elevation": elev_display,
                "type": "station"
            }
            features.append(Feature(geometry=point, properties=properties))
        except Exception as e:
            logging.warning(f"No se pudo exportar {est.nombre} a GeoJSON: {e}")

    # Crear y guardar FeatureCollection
    try:
        collection = FeatureCollection(features, properties={"author": f"{author} - Perfil altimétrico para vías férreas"})
        with open(archivo_geojson, 'w', encoding='utf-8') as f:
            dump(collection, f, indent=2)
        logging.info(f"GeoJSON guardado en: {archivo_geojson}")
    except Exception as e:
        logging.error(f"Error al guardar GeoJSON: {e}")

def exportar_pdf(fig: Optional[go.Figure], archivo_pdf: str):
    """Exporta figura a PDF."""
    if fig is None:
        logging.warning("No hay figura para PDF")
        return

    try:
        fig.write_image(archivo_pdf, width=1600, height=700, engine="kaleido")
        logging.info(f"PDF guardado en: {archivo_pdf}")
    except Exception as e:
        logging.error(f"Error al exportar PDF: {e}")

def exportar_csv(puntos_con_elevacion: List[InterpolatedPoint], slope_data: Optional[np.ndarray], archivo_csv: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta puntos interpolados a CSV con atribución."""
    try:
        with open(archivo_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"# {author} - Perfil altimétrico para vías férreas"])
            writer.writerow(["Km", "Lat", "Lon", "Elevation", "Slope_m_per_km"])
            for i, p in enumerate(puntos_con_elevacion):
                slope = slope_data[i] if slope_data is not None and i < len(slope_data) and not np.isnan(slope_data[i]) else ""
                writer.writerow([p.km, p.lat, p.lon, p.elevation if p.elevation is not None else "", slope])
        logging.info(f"CSV guardado en: {archivo_csv}")
    except Exception as e:
        logging.error(f"Error al guardar CSV: {e}")

def seleccionar_grupos(estaciones: List[Station], tramos: Optional[List[str]] = None) -> List[List[Station]]:
    """Selecciona tramos interactivamente o vía CLI."""
    if not estaciones:
        logging.error("No hay estaciones")
        return []

    if tramos:
        grupos = []
        for tramo in tramos:
            try:
                ini, fin = map(int, tramo.split(':'))
                if not (0 <= ini <= fin < len(estaciones)):
                    logging.error(f"Tramo inválido: {tramo}")
                    continue
                grupos.append(estaciones[ini:fin + 1])
            except ValueError:
                logging.error(f"Formato de tramo inválido: {tramo}")
        if grupos:
            return grupos
        logging.warning("No se parsearon tramos, usando interactivo")

    print("\n--- Selección de Tramos ---")
    print("Índice de estaciones:")
    for i, est in enumerate(estaciones):
        print(f"  {i:>2}: {est.nombre} (Km {est.km:.3f})")

    grupos_seleccionados = []
    try:
        resp = input("\n¿Dividir en tramos? (s/n) [n]: ").lower().strip() or 'n'
        if resp == 's':
            while True:
                try:
                    ini = int(input(f"Índice inicial (0-{len(estaciones)-1}): "))
                    if not 0 <= ini < len(estaciones):
                        print(f"Índice inválido")
                        continue
                    fin = int(input(f"Índice final ({ini}-{len(estaciones)-1}): "))
                    if not ini <= fin < len(estaciones):
                        print(f"Índice inválido")
                        continue
                    tramo = estaciones[ini:fin + 1]
                    print(f"  -> Tramo añadido: {tramo[0].nombre} a {tramo[-1].nombre}")
                    grupos_seleccionados.append(tramo)
                    otro = input("¿Otro tramo? (s/n) [n]: ").lower().strip() or 'n'
                    if otro != 's':
                        break
                except ValueError:
                    print("Introduce un número válido")
                except EOFError:
                    print("\nEntrada interrumpida")
                    return []
        else:
            grupos_seleccionados = [estaciones]
    except EOFError:
        print("\nEntrada interrumpida")
        return []

    return [g for g in grupos_seleccionados if g]

def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Generador de perfiles altimétricos para vías férreas")
    parser.add_argument("--csv", default=DEFAULT_STATIONS_FILE, help="Archivo CSV de estaciones")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_METERS, help="Intervalo de interpolación (m)")
    parser.add_argument("--tramos", nargs="*", help="Tramos como 'inicio:fin'")
    parser.add_argument("--no-pdf", action="store_true", help="No exportar PDF")
    parser.add_argument("--kml", action="store_true", help="Exportar KML")
    parser.add_argument("--csv-out", action="store_true", help="Exportar CSV")
    parser.add_argument("--geojson", action="store_true", help="Exportar GeoJSON")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Ventana para suavizado de pendiente")
    parser.add_argument("--plot-theme", choices=["light", "dark"], default="light", help="Tema del gráfico (light, dark)")
    parser.add_argument("--colors", default="blue,orange", help="Colores para elevación,pendiente (ej: blue,red)")
    parser.add_argument("--watermark", default="LAL", help="Texto de marca de agua (o 'none' para desactivar)")
    parser.add_argument("--author", default=AUTHOR_ATTRIBUTION, help="Texto de atribución del autor")
    return parser.parse_args()

def main():
    """Función principal."""
    args = parse_args()
    logging.info(f">>> Iniciando script - {args.author} <<<")
    print(f"\n{args.author} - Perfil altimétrico para vías férreas")

    try:
        estaciones_todas = cargar_estaciones(args.csv)
        grupos_de_estaciones = seleccionar_grupos(estaciones_todas, args.tramos)

        if not grupos_de_estaciones:
            logging.warning("No hay tramos para procesar")
            print("No hay tramos para procesar")
            return

        for i, tramo_actual in enumerate(grupos_de_estaciones):
            if len(tramo_actual) == len(estaciones_todas) and len(grupos_de_estaciones) == 1:
                nombre_tramo = "Ruta Completa"
            else:
                nombre_tramo = f"Tramo {i+1}: {tramo_actual[0].nombre} a {tramo_actual[-1].nombre}"

            logging.info(f"\n--- Procesando {nombre_tramo} [{len(tramo_actual)} estaciones] ---")

            if len(tramo_actual) < 2:
                logging.warning(f"{nombre_tramo} tiene menos de 2 estaciones")
                print(f"Advertencia: {nombre_tramo} omitido (mínimo 2 estaciones)")
                continue

            print(f"\nProcesando: {nombre_tramo}")

            default_title = f"Perfil Altimétrico - {tramo_actual[0].nombre} a {tramo_actual[-1].nombre}"
            try:
                print(f"\nTítulo del gráfico (dejar vacío para '{default_title}'):")
                titulo_final = input("Título: ").strip()
                if not titulo_final:
                    titulo_final = default_title
                print(f"  Título seleccionado: '{titulo_final}'")
            except EOFError:
                print("Entrada interrumpida, usando título por defecto")
                titulo_final = default_title

            puntos_interp = interpolar_puntos(tramo_actual, args.interval)
            if not puntos_interp:
                logging.error(f"Fallo en interpolación para {nombre_tramo}")
                continue

            puntos_con_elev = obtener_elevaciones_paralelo(puntos_interp, author=args.author)

            kms_array = np.array([p.km for p in puntos_con_elev])
            elevs_array = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elev])
            slope_m_per_km = calcular_pendiente_suavizada(kms_array, elevs_array, args.smooth_window)

            valid_elevs = elevs_array[~np.isnan(elevs_array)]
            if len(valid_elevs) > 1:
                elevation_diffs = np.diff(valid_elevs)
                ascent = np.sum(elevation_diffs[elevation_diffs > 0])
                descent = -np.sum(elevation_diffs[elevation_diffs < 0])
            else:
                ascent = descent = 0.0

            valid_slopes = slope_m_per_km[~np.isnan(slope_m_per_km)]
            if len(valid_slopes) > 0:
                max_uphill = np.max(valid_slopes[valid_slopes > 0]) if np.any(valid_slopes > 0) else 0
                max_downhill = np.min(valid_slopes[valid_slopes < 0]) if np.any(valid_slopes < 0) else 0
                avg_abs_slope = np.mean(np.abs(valid_slopes))
                print(f"  Resumen de Pendiente:")
                print(f"    - Máxima subida: {max_uphill:.1f} m/km")
                print(f"    - Máxima bajada: {max_downhill:.1f} m/km")
                print(f"    - Media (abs): {avg_abs_slope:.1f} m/km")
                print(f"  Desnivel acumulado:")
                print(f"    - Ascenso total: {ascent:.1f} m")
                print(f"    - Descenso total: {descent:.1f} m")
            else:
                print("  No se pudo calcular pendiente ni desnivel")

            try:
                default_name = get_next_output_name(os.getcwd(), DEFAULT_OUTPUT_PREFIX)
                print(f"\nNombre base para archivos (dejar vacío para '{default_name}'):")
                nombre_base = input("Nombre: ").strip()
                if not nombre_base:
                    nombre_base = default_name
                else:
                    nombre_base = sanitize_filename(nombre_base)
                    if not nombre_base:
                        logging.warning("Nombre inválido, usando nombre por defecto")
                        nombre_base = default_name
                print(f"  Archivos base: '{nombre_base}'")
            except EOFError:
                print("Entrada interrumpida, usando nombre por defecto")
                nombre_base = default_name

            archivo_html = f"{nombre_base}.html"
            figura_plotly = graficar_html(puntos_con_elev, tramo_actual, archivo_html,
                                          titulo=titulo_final, slope_data=slope_m_per_km,
                                          theme=args.plot_theme, colors=args.colors,
                                          watermark=args.watermark)

            if not args.no_pdf:
                try:
                    resp_pdf = input(f"¿Exportar '{titulo_final}' a PDF? (s/n) [n]: ").lower().strip() or 'n'
                    if resp_pdf == 's':
                        archivo_pdf = f"{nombre_base}.pdf"
                        exportar_pdf(figura_plotly, archivo_pdf)
                except EOFError:
                    print("Entrada interrumpida")

            if args.kml or (not args.tramos and input(f"¿Exportar KML? (s/n) [n]: ").lower().strip() == 's'):
                archivo_kml = f"{nombre_base}_estaciones.kml"
                exportar_kml(puntos_con_elev, tramo_actual, archivo_kml, author=args.author)

            if args.csv_out or (not args.tramos and input(f"¿Exportar CSV? (s/n) [n]: ").lower().strip() == 's'):
                archivo_csv = f"{nombre_base}_datos.csv"
                exportar_csv(puntos_con_elev, slope_m_per_km, archivo_csv, author=args.author)

            if args.geojson or (not args.tramos and input(f"¿Exportar GeoJSON? (s/n) [n]: ").lower().strip() == 's'):
                archivo_geojson = f"{nombre_base}_datos.geojson"
                exportar_geojson(puntos_con_elev, tramo_actual, archivo_geojson, author=args.author)

        logging.info(">>> Proceso completado <<<")
        print("\nProceso finalizado")

    except FileNotFoundError:
        print(f"\nError: No se encontró '{args.csv}'")
    except ValueError as e:
        print(f"\nError en datos: {e}")
    except Exception as e:
        logging.exception("Error inesperado")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
