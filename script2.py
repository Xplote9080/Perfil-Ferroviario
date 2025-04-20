# -*- coding: utf-8 -*-
"""
Script para generar perfiles altimétricos a partir de datos de estaciones para vías férreas.
© 2025 LAL - Todos los derechos reservados.

Funcionalidades:
- Carga estaciones desde un archivo CSV.
- Interpola puntos usando CubicSpline.
- Obtiene elevaciones con la API de Open-Meteo (con caché en CSV y llamadas por lotes paralelas).
- Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay.
- Calcula desnivel acumulado (ascenso/descenso total).
- Genera gráficos HTML interactivos con Plotly (elevación + pendiente) con marca de agua.
- Exporta a PDF (opcional, requiere 'kaleido'), KML, CSV y GeoJSON.
- Valida rangos de latitud, longitud y kilómetros.
- Incluye atribución en todos los outputs: "Perfil altimétrico ferroviario - LAL 2025".
- Optimizado para Streamlit Cloud con progreso visual y manejo de archivos temporales.
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
import tempfile
import pandas as pd
from typing import List, Tuple, Optional, NamedTuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed
import sys

# --- Configuración y Constantes ---
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CACHE_FILE = "elevations_cache.csv"
DEFAULT_STATIONS_FILE = "estaciones.csv"
DEFAULT_INTERVAL_METERS = 200  # Ajustado a 200 m para mejor rendimiento en Streamlit
REQUEST_TIMEOUT_SECONDS = 15
MAX_API_WORKERS = 4  # Respetando tu límite
MAX_BATCH_SIZE = 900  # Tamaño del lote para la API (Open-Meteo permite hasta 1000)
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

# --- Caché en memoria ---
_cache = None

def load_cache_to_memory() -> Dict[Tuple[float, float], float]:
    """Carga el caché de elevaciones en memoria desde el CSV."""
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    if not os.path.exists(ELEVATION_CACHE_FILE):
        return _cache
    try:
        with open(ELEVATION_CACHE_FILE, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#') or row[0].lower() == 'latitude':
                    continue
                try:
                    lat, lon, elev = map(float, row[:3])
                    _cache[(lat, lon)] = elev
                except ValueError:
                    logging.warning(f"Fila inválida en caché CSV: {row}")
    except Exception as e:
        logging.warning(f"Error al leer caché CSV: {e}")
    return _cache

def _load_elevation_from_cache(lat: float, lon: float) -> Optional[float]:
    """Carga elevación desde el caché en memoria."""
    cache = load_cache_to_memory()
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    return cache.get((lat_r, lon_r))

def _save_elevation_to_cache(lat: float, lon: float, elevation: float, author: str = AUTHOR_ATTRIBUTION):
    """Guarda elevación en el archivo CSV y en memoria."""
    global _cache
    cache = load_cache_to_memory()
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    if (lat_r, lon_r) in cache:
        return  # Evitar duplicados
    cache[(lat_r, lon_r)] = elevation
    try:
        file_exists = os.path.exists(ELEVATION_CACHE_FILE)
        with open(ELEVATION_CACHE_FILE, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(ELEVATION_CACHE_FILE) == 0:
                writer.writerow([f"# {author} - Caché de elevaciones para vías férreas"])
                writer.writerow(["latitude", "longitude", "elevation"])
            writer.writerow([lat_r, lon_r, elevation])
    except Exception as e:
        logging.warning(f"Error al guardar en caché CSV: {e}")

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

# --- Funciones principales ---
def cargar_estaciones(archivo_csv: str) -> List[Station]:
    """Carga estaciones desde un archivo CSV."""
    try:
        df = pd.read_csv(archivo_csv)
        df.columns = [c.strip().capitalize() for c in df.columns]
        required_columns = {'Nombre', 'Km', 'Lat', 'Lon'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"El CSV debe contener las columnas: {required_columns}")
        
        estaciones = []
        for _, row in df.iterrows():
            lat = float(row['Lat'])
            lon = float(row['Lon'])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Coordenadas inválidas en {row['Nombre']}: Lat={lat}, Lon={lon}")
            estaciones.append(Station(
                nombre=str(row['Nombre']).strip(),
                km=float(row['Km']),
                lat=lat,
                lon=lon
            ))
        
        # Validar orden de kilómetros
        kms = [s.km for s in estaciones]
        if len(kms) < 2:
            raise ValueError("Se requieren al menos 2 estaciones")
        if not all(kms[i] < kms[i+1] for i in range(len(kms)-1)):
            raise ValueError("Los kilómetros deben ser estrictamente crecientes")
        
        logging.info(f"Cargadas {len(estaciones)} estaciones desde {archivo_csv}")
        return estaciones
    except Exception as e:
        logging.error(f"Error al cargar estaciones: {e}")
        raise

def interpolar_puntos(estaciones: List[Station], intervalo_m: int = DEFAULT_INTERVAL_METERS) -> List[InterpolatedPoint]:
    """Interpola puntos geográficos."""
    if len(estaciones) < 2:
        logging.warning("Mínimo 2 estaciones requeridas para interpolación")
        return []

    kms = np.array([s.km for s in estaciones])
    lats = np.array([s.lat for s in estaciones])
    lons = np.array([s.lon for s in estaciones])

    unique_kms, unique_indices = np.unique(kms, return_index=True)
    if len(unique_kms) < 2:
        logging.error("No hay suficientes puntos con Km únicos y crecientes")
        return []

    valid_kms = kms[unique_indices]
    valid_lats = lats[unique_indices]
    valid_lons = lons[unique_indices]

    try:
        spline_lat = CubicSpline(valid_kms, valid_lats)
        spline_lon = CubicSpline(valid_kms, valid_lons)
    except ValueError as e:
        logging.error(f"Error al crear splines cúbicas: {e}")
        return []

    km_inicio = valid_kms[0]
    km_fin = valid_kms[-1]
    if km_fin <= km_inicio:
        logging.error("Km final no es mayor que Km inicial")
        return []

    distancia_km = km_fin - km_inicio
    num_puntos = max(2, int(np.ceil(distancia_km * 1000 / intervalo_m)) + 1)
    kms_interp = np.linspace(km_inicio, km_fin, num_puntos)
    kms_interp = kms_interp[kms_interp >= km_inicio]
    kms_interp = kms_interp[kms_interp <= km_fin]
    if kms_interp[-1] < km_fin:
        kms_interp = np.append(kms_interp, km_fin)

    lats_interp = spline_lat(kms_interp)
    lons_interp = spline_lon(kms_interp)

    puntos = [InterpolatedPoint(km=float(km_i), lat=float(lat_i), lon=float(lon_i))
              for km_i, lat_i, lon_i in zip(kms_interp, lats_interp, lons_interp)]

    puntos = [p for p in puntos if -90 <= p.lat <= 90 and -180 <= p.lon <= 180]
    logging.info(f"Interpolados {len(puntos)} puntos cada {intervalo_m} m entre Km {km_inicio:.3f} y {km_fin:.3f}")
    return puntos

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def _fetch_elevation_batch_from_api(batch_coords: List[Tuple[float, float]], timeout: int = REQUEST_TIMEOUT_SECONDS) -> List[float]:
    """Obtiene elevaciones para un lote de coordenadas desde la API de Open-Meteo."""
    if not batch_coords:
        return []

    lats = [f"{c[0]:.5f}" for c in batch_coords]
    lons = [f"{c[1]:.5f}" for c in batch_coords]
    params = {'latitude': lats, 'longitude': lons}

    try:
        response = requests.get(ELEVATION_API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        elevations = data.get("elevation", [])
        if not isinstance(elevations, list) or len(elevations) != len(batch_coords):
            raise requests.exceptions.RequestException(f"Expected {len(batch_coords)} elevations, got {len(elevations)}")
        return [float(e) if isinstance(e, (int, float)) else DEFAULT_ELEVATION_ON_ERROR for e in elevations]
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for batch (first point: {lats[0]}, {lons[0]}): {e}")
        raise

def obtener_elevaciones_paralelo(puntos: List[InterpolatedPoint], author: str = AUTHOR_ATTRIBUTION, progress_callback: Optional[callable] = None) -> List[InterpolatedPoint]:
    """Obtiene elevaciones usando caché y API por lotes en paralelo."""
    if not puntos:
        return []

    logging.info(f"Iniciando consulta de elevaciones para {len(puntos)} puntos...")
    puntos_con_elevacion = [None] * len(puntos)
    points_to_fetch_details = []

    logging.info("Paso 1/2: Buscando elevaciones en caché...")
    for i, punto in enumerate(puntos):
        cached_elev = _load_elevation_from_cache(punto.lat, punto.lon)
        if cached_elev is not None:
            puntos_con_elevacion[i] = punto._replace(elevation=cached_elev)
        else:
            points_to_fetch_details.append((i, punto.lat, punto.lon))
        if progress_callback:
            progress_callback((i + 1) / (len(puntos) * 2))  # Primera mitad del progreso

    num_cached = len(puntos) - len(points_to_fetch_details)
    logging.info(f"{num_cached} puntos encontrados en caché.")

    if not points_to_fetch_details:
        logging.info("Todos los puntos encontrados en caché.")
        return puntos_con_elevacion

    logging.info(f"Paso 2/2: Obteniendo {len(points_to_fetch_details)} elevaciones desde API...")
    batches_data = []
    for i in range(0, len(points_to_fetch_details), MAX_BATCH_SIZE):
        batch_details = points_to_fetch_details[i:i + MAX_BATCH_SIZE]
        batch_coords_only = [(detail[1], detail[2]) for detail in batch_details]
        batch_original_indices = [detail[0] for detail in batch_details]
        batches_data.append((batch_coords_only, batch_original_indices))

    futures = {}
    start_time = time.monotonic()
    processed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        for batch_coords, original_indices_batch in batches_data:
            future = executor.submit(_fetch_elevation_batch_from_api, batch_coords, REQUEST_TIMEOUT_SECONDS)
            futures[future] = (batch_coords, original_indices_batch)

        num_batch_failures = 0
        for future in as_completed(futures):
            batch_coords_used, original_indices_batch = futures[future]
            batch_size = len(original_indices_batch)
            try:
                elevations_from_batch = future.result()
                for i_in_batch, original_idx in enumerate(original_indices_batch):
                    original_point = puntos[original_idx]
                    puntos_con_elevacion[original_idx] = original_point._replace(elevation=elevations_from_batch[i_in_batch])
                    _save_elevation_to_cache(original_point.lat, original_point.lon, elevations_from_batch[i_in_batch], author)
                processed_count += batch_size
            except Exception as e:
                logging.error(f"Error en lote de {batch_size} puntos: {e}")
                num_batch_failures += 1
                for original_idx in original_indices_batch:
                    original_point = puntos[original_idx]
                    puntos_con_elevacion[original_idx] = original_point._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)
                processed_count += batch_size
            if progress_callback:
                progress_callback(0.5 + (processed_count / len(points_to_fetch_details)) / 2)  # Segunda mitad del progreso

    end_time = time.monotonic()
    logging.info(f"Procesamiento completado en {end_time - start_time:.2f}s.")
    if num_batch_failures > 0:
        logging.warning(f"{num_batch_failures} lotes fallaron.")

    for i in range(len(puntos_con_elevacion)):
        if puntos_con_elevacion[i] is None or puntos_con_elevacion[i].elevation is None:
            logging.warning(f"Punto en índice {i} sin elevación. Asignando por defecto.")
            puntos_con_elevacion[i] = puntos[i]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)

    return puntos_con_elevacion

def calcular_pendiente_suavizada(kms: np.ndarray, elevs: np.ndarray, window_length: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay."""
    slope_m_per_km = np.full_like(elevs, np.nan)
    kms_m = kms * 1000.0
    valid_indices = ~np.isnan(elevs)
    if np.count_nonzero(valid_indices) < 3 or window_length < 3:
        logging.warning("No hay suficientes datos válidos para calcular pendiente")
        return slope_m_per_km

    valid_kms_m = kms_m[valid_indices]
    valid_elevs = elevs[valid_indices]
    try:
        window = min(window_length, len(valid_elevs))
        if window % 2 == 0:
            window += 1
        if window < 3:
            window = 3
        if window > len(valid_elevs):
            window = len(valid_elevs)
            if window % 2 == 0:
                window -= 1
            if window < 3:
                logging.warning("Puntos válidos insuficientes para Savitzky-Golay.")
                return slope_m_per_km

        polyorder = min(2, window - 1)
        if polyorder < 1:
            logging.warning("Grado de polinomio muy bajo.")
            return slope_m_per_km

        gradient_values = savgol_filter(valid_elevs, window, polyorder, deriv=1, delta=np.diff(valid_kms_m)[0] if len(valid_kms_m) > 1 else 1)
        slope_m_per_km_values = gradient_values * 1000.0
        slope_m_per_km[valid_indices] = slope_m_per_km_values
    except Exception as e:
        logging.error(f"Error al calcular pendiente suavizada: {e}")
        return np.full_like(elevs, np.nan)
    return slope_m_per_km

def graficar_html(puntos_con_elevacion: List[InterpolatedPoint],
                  estaciones_tramo: List[Station],
                  archivo_html: str,
                  titulo: str = "Perfil altimétrico",
                  slope_data: Optional[np.ndarray] = None,
                  theme: str = "light",
                  colors: str = "blue,orange",
                  watermark: str = "LAL") -> Optional[go.Figure]:
    """Genera gráfico interactivo con tema, colores y marca de agua."""
    if not puntos_con_elevacion:
        logging.warning("No hay puntos para graficar")
        return None

    kms = np.array([p.km for p in puntos_con_elevacion])
    elevs = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elevacion], dtype=float)

    try:
        elev_color, slope_color = colors.split(',')
    except ValueError:
        logging.warning("Formato de colores inválido, usando predeterminados")
        elev_color, slope_color = "blue", "orange"

    has_slope_data = slope_data is not None and len(slope_data) == len(kms)
    hover_texts = []
    for i, p in enumerate(puntos_con_elevacion):
        elev_text = f"{p.elevation:.1f}" if p.elevation is not None else "N/A"
        slope_text = "N/A" if not has_slope_data or np.isnan(slope_data[i]) else f"{slope_data[i]:.1f}"
        hover_texts.append(f"<b>Km: {p.km:.3f}</b><br>Elev: {elev_text} m<br>Pendiente: {slope_text} m/km")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=kms, y=elevs, mode='lines', name='Elevación',
        line=dict(color=elev_color, width=2),
        hoverinfo='text', text=hover_texts, yaxis='y1',
        customdata=np.c_[kms, elevs, slope_data if has_slope_data else np.full_like(kms, np.nan)],
        hoverlabel=dict(namelength=-1)
    ))

    if estaciones_tramo:
        station_kms = np.array([s.km for s in estaciones_tramo])
        station_names = [s.nombre for s in estaciones_tramo]
        indices_closest = np.searchsorted(kms, station_kms)
        indices_closest = np.clip(indices_closest, 0, len(kms) - 1)

        for i, est in enumerate(estaciones_tramo):
            idx = indices_closest[i]
            elev_display = elevs[idx] if not np.isnan(elevs[idx]) else DEFAULT_ELEVATION_ON_ERROR
            slope_text = "N/A" if not has_slope_data or np.isnan(slope_data[idx]) else f"{slope_data[idx]:.1f}"
            fig.add_trace(go.Scatter(
                x=[kms[idx]], y=[elev_display],
                mode='markers+text', text=[est.nombre],
                textposition="top center",
                marker=dict(size=10, color='red', symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey')),
                name=est.nombre,
                hoverinfo='text',
                hovertext=f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Elev: {elev_display:.1f} m<br>Pendiente: {slope_text} m/km",
                yaxis='y1'
            ))

    if has_slope_data:
        fig.add_trace(go.Scattergl(
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
            font=dict(size=80, color="rgba(150, 150, 150, 0.3)"),
            textangle=-45,
            opacity=0.3,
            xanchor='center', yanchor='middle'
        ))

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center'),
        xaxis_title="Kilómetro",
        yaxis=dict(
            title=dict(text="Elevación (msnm)", font=dict(color=elev_color)),
            tickfont=dict(color=elev_color),
            hoverformat=".1f"
        ),
        yaxis2=dict(
            title=dict(text="Pendiente (m/km)", font=dict(color=slope_color)),
            tickfont=dict(color=slope_color),
            anchor="x", overlaying="y", side="right", showgrid=False,
            hoverformat=".1f"
        ),
        xaxis=dict(hoverformat='.3f'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=50),
        template=template,
        annotations=annotations,
        newselection=dict(mode="reset")
    )

    try:
        plot(fig, filename=archivo_html, auto_open=False, include_plotlyjs='cdn')
        logging.info(f"Gráfico guardado en: {archivo_html}")
    except Exception as e:
        logging.error(f"Error al guardar HTML: {e}")
        return None
    return fig

def exportar_kml(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], archivo_kml: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta estaciones a KML."""
    kml = simplekml.Kml()
    for est in estaciones_tramo:
        closest_point = min(puntos_con_elevacion, key=lambda p: abs(p.km - est.km), default=None)
        elev = closest_point.elevation if closest_point and closest_point.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        pnt = kml.newpoint(name=est.nombre, coords=[(est.lon, est.lat, elev)])
        pnt.description = f"Km: {est.km:.3f}, Elevación: {elev:.1f} m\n{author}"
    try:
        kml.save(archivo_kml)
        logging.info(f"KML guardado en: {archivo_kml}")
    except Exception as e:
        logging.error(f"Error al guardar KML: {e}")

def exportar_geojson(puntos_con_elevacion: List[InterpolatedPoint], estaciones_tramo: List[Station], archivo_geojson: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta puntos y estaciones a GeoJSON."""
    features = []
    for p in puntos_con_elevacion:
        elev = p.elevation if p.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        features.append(Feature(
            geometry=Point((p.lon, p.lat)),
            properties={"km": p.km, "elevation": elev, "type": "interpolated", "author": author}
        ))
    for est in estaciones_tramo:
        closest_point = min(puntos_con_elevacion, key=lambda p: abs(p.km - est.km), default=None)
        elev = closest_point.elevation if closest_point and closest_point.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
        features.append(Feature(
            geometry=Point((est.lon, est.lat)),
            properties={"name": est.nombre, "km": est.km, "elevation": elev, "type": "station", "author": author}
        ))
    collection = FeatureCollection(features)
    try:
        with open(archivo_geojson, 'w', encoding='utf-8') as f:
            dump(collection, f, indent=2)
        logging.info(f"GeoJSON guardado en: {archivo_geojson}")
    except Exception as e:
        logging.error(f"Error al guardar GeoJSON: {e}")

def exportar_pdf(fig: go.Figure, archivo_pdf: str):
    """Exporta gráfico a PDF usando kaleido."""
    try:
        import kaleido
        fig.write_image(archivo_pdf, engine="kaleido")
        logging.info(f"PDF guardado en: {archivo_pdf}")
    except ImportError:
        logging.warning("Kaleido no está instalado. Exportación a PDF omitida.")
    except Exception as e:
        logging.error(f"Error al guardar PDF: {e}")

def exportar_csv(puntos_con_elevacion: List[InterpolatedPoint], slope_data: np.ndarray, archivo_csv: str, author: str = AUTHOR_ATTRIBUTION):
    """Exporta datos interpolados a CSV."""
    try:
        with open(archivo_csv, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"# {author}"])
            writer.writerow(["km", "latitude", "longitude", "elevation", "slope_m_per_km"])
            for i, p in enumerate(puntos_con_elevacion):
                elev = p.elevation if p.elevation is not None else DEFAULT_ELEVATION_ON_ERROR
                slope = slope_data[i] if i < len(slope_data) and not np.isnan(slope_data[i]) else None
                writer.writerow([p.km, p.lat, p.lon, elev, slope])
        logging.info(f"CSV guardado en: {archivo_csv}")
    except Exception as e:
        logging.error(f"Error al guardar CSV: {e}")

def seleccionar_grupos(estaciones: List[Station], tramos: Optional[List[str]] = None) -> List[List[Station]]:
    """Selecciona grupos de estaciones para CLI (no usado en Streamlit)."""
    if not sys.stdin.isatty():  # No interactivo (Streamlit)
        return [estaciones] if len(estaciones) >= 2 else []
    
    if tramos:
        grupos = []
        for tramo in tramos:
            try:
                inicio, fin = tramo.split('-')
                inicio, fin = inicio.strip(), fin.strip()
                grupo = [s for s in estaciones if s.nombre in [inicio, fin]]
                if len(grupo) == 2:
                    grupos.append(sorted(grupo, key=lambda s: s.km))
                else:
                    logging.warning(f"Tramo inválido: {tramo}")
            except ValueError:
                logging.warning(f"Formato de tramo inválido: {tramo}")
        return grupos

    print("\nEstaciones disponibles:")
    for i, est in enumerate(estaciones):
        print(f"{i+1}. {est.nombre} (Km {est.km:.3f})")
    grupos = []
    while True:
        try:
            resp = input("\nIngresa tramo (ej: 1-3) o presiona Enter para usar todas: ").strip()
            if not resp:
                return [estaciones]
            inicio, fin = map(int, resp.split('-'))
            if 1 <= inicio <= len(estaciones) and 1 <= fin <= len(estaciones) and inicio != fin:
                grupo = [estaciones[inicio-1], estaciones[fin-1]]
                grupos.append(sorted(grupo, key=lambda s: s.km))
            else:
                print("Índices inválidos.")
        except ValueError:
            print("Formato inválido. Usa 'inicio-fin' (ej: 1-3).")
        if input("¿Agregar otro tramo? (s/n) [n]: ").lower().strip() != 's':
            break
    return grupos

# --- CLI (mantenido para compatibilidad, pero no usado en Streamlit) ---
def parse_args():
    """Parses command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Generador de Perfil Altimétrico Ferroviario")
    parser.add_argument('--csv', default=DEFAULT_STATIONS_FILE, help="Archivo CSV de estaciones")
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_METERS, help="Intervalo de interpolación (metros)")
    parser.add_argument('--smooth-window', type=int, default=DEFAULT_SMOOTH_WINDOW, help="Ventana de suavizado")
    parser.add_argument('--tramos', nargs='*', help="Tramos (ej: Estacion1-Estacion2)")
    parser.add_argument('--plot-theme', default='light', choices=['light', 'dark'], help="Tema del gráfico")
    parser.add_argument('--colors', default='blue,orange', help="Colores para elevación,pendiente")
    parser.add_argument('--watermark', default='LAL', help="Marca de agua")
    parser.add_argument('--no-pdf', action='store_true', help="No exportar a PDF")
    parser.add_argument('--kml', action='store_true', help="Exportar a KML")
    parser.add_argument('--csv-out', action='store_true', help="Exportar a CSV")
    parser.add_argument('--geojson', action='store_true', help="Exportar a GeoJSON")
    parser.add_argument('--author', default=AUTHOR_ATTRIBUTION, help="Autor para atribución")
    return parser.parse_args()

def main():
    """Función principal para CLI."""
    args = parse_args()
    logging.info(f">>> Iniciando script - {args.author} <<<")
    print(f"\n{args.author} - Generador de Perfil Altimétrico Ferroviario")

    try:
        estaciones_todas = cargar_estaciones(args.csv)
        grupos_de_estaciones = seleccionar_grupos(estaciones_todas, args.tramos)
        if not grupos_de_estaciones:
            logging.warning("No hay tramos para procesar")
            print("No hay tramos para procesar.")
            return

        for i, tramo_actual in enumerate(grupos_de_estaciones):
            nombre_tramo = f"Tramo {i+1}: {tramo_actual[0].nombre} a {tramo_actual[-1].nombre}" if len(tramo_actual) >= 2 else "Ruta Completa"
            logging.info(f"\n--- Procesando {nombre_tramo} [{len(tramo_actual)} estaciones] ---")
            print(f"\nProcesando: {nombre_tramo}")

            default_title = f"Perfil Altimétrico - {tramo_actual[0].nombre} a {tramo_actual[-1].nombre}"
            titulo_final = default_title
            if not args.tramos:
                try:
                    print(f"\nTítulo del gráfico (dejar vacío para '{default_title}'):")
                    titulo_input = input("Título: ").strip()
                    if titulo_input:
                        titulo_final = titulo_input
                    print(f"  Título seleccionado: '{titulo_final}'")
                except EOFError:
                    print("\nEntrada de título interrumpida, usando título por defecto.")
                except Exception as e:
                    logging.warning(f"Error al leer título: {e}, usando por defecto.")

            puntos_interp = interpolar_puntos(tramo_actual, args.interval)
            if not puntos_interp:
                logging.error(f"Fallo en interpolación para {nombre_tramo}")
                continue

            puntos_con_elev = obtener_elevaciones_paralelo(puntos_interp, author=args.author)
            kms_array = np.array([p.km for p in puntos_con_elev])
            elevs_array = np.array([p.elevation if p.elevation is not None else np.nan for p in puntos_con_elev])

            valid_elev_indices = ~np.isnan(elevs_array)
            valid_elevs_for_stats = elevs_array[valid_elev_indices]
            valid_kms_for_stats = kms_array[valid_elev_indices]

            if len(valid_elevs_for_stats) > 1:
                elevation_diffs = np.diff(valid_elevs_for_stats)
                ascent = np.sum(elevation_diffs[elevation_diffs > 0])
                descent = -np.sum(elevation_diffs[elevation_diffs < 0])
                slope_m_per_km = calcular_pendiente_suavizada(kms_array, elevs_array, args.smooth_window)
                valid_slopes_indices = ~np.isnan(slope_m_per_km)
                valid_slopes_for_stats = slope_m_per_km[valid_slopes_indices]

                if len(valid_slopes_for_stats) > 0:
                    max_uphill = np.max(valid_slopes_for_stats[valid_slopes_for_stats > 0]) if np.any(valid_slopes_for_stats > 0) else 0.0
                    max_downhill = np.min(valid_slopes_for_stats[valid_slopes_for_stats < 0]) if np.any(valid_slopes_for_stats < 0) else 0.0
                    avg_abs_slope = np.mean(np.abs(valid_slopes_for_stats))
                    print(f"  Resumen de Pendiente:")
                    print(f"    - Máxima subida: {max_uphill:.1f} m/km")
                    print(f"    - Máxima bajada: {max_downhill:.1f} m/km")
                    print(f"    - Media (abs): {avg_abs_slope:.1f} m/km")
                    print(f"  Desnivel acumulado:")
                    print(f"    - Ascenso total: {ascent:.1f} m")
                    print(f"    - Descenso total: {descent:.1f} m")
                else:
                    logging.warning("No se pudieron calcular estadísticas de pendiente.")
            else:
                slope_m_per_km = np.full_like(elevs_array, np.nan)
                logging.warning("Datos de elevación insuficientes.")

            default_name = get_next_output_name(os.getcwd(), DEFAULT_OUTPUT_PREFIX)
            nombre_base = default_name
            if not args.tramos:
                try:
                    print(f"\nNombre base para archivos (dejar vacío para '{default_name}'):")
                    nombre_base_input = input("Nombre: ").strip()
                    if nombre_base_input:
                        nombre_base = sanitize_filename(nombre_base_input) or default_name
                    print(f"  Archivos base: '{nombre_base}'")
                except EOFError:
                    print("\nEntrada de nombre base interrumpida, usando por defecto.")
                except Exception as e:
                    logging.warning(f"Error al leer nombre base: {e}, usando por defecto.")

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                archivo_html = tmp.name
                figura_plotly = graficar_html(
                    puntos_con_elevacion=puntos_con_elev, estaciones_tramo=tramo_actual,
                    archivo_html=archivo_html, titulo=titulo_final, slope_data=slope_m_per_km,
                    theme=args.plot_theme, colors=args.colors, watermark=args.watermark
                )

            archivos_temporales = [archivo_html]
            if not args.no_pdf and figura_plotly:
                export_pdf_q = True
                if not args.tramos:
                    try:
                        resp_pdf = input(f"¿Exportar '{titulo_final}' a PDF? (s/n) [s]: ").lower().strip() or 's'
                        export_pdf_q = (resp_pdf == 's')
                    except EOFError:
                        print("\nEntrada de exportar PDF interrumpida, usando por defecto (sí).")
                    except Exception as e:
                        logging.warning(f"Error al leer opción PDF: {e}, usando por defecto (sí).")
                if export_pdf_q:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        archivo_pdf = tmp.name
                        exportar_pdf(figura_plotly, archivo_pdf)
                        archivos_temporales.append(archivo_pdf)

            if args.kml or (not args.tramos and input(f"¿Exportar KML? (s/n) [n]: ").lower().strip() == 's'):
                with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
                    archivo_kml = tmp.name
                    exportar_kml(puntos_con_elev, tramo_actual, archivo_kml, author=args.author)
                    archivos_temporales.append(archivo_kml)

            if args.csv_out or (not args.tramos and input(f"¿Exportar CSV? (s/n) [s]: ").lower().strip() == 's'):
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    archivo_csv = tmp.name
                    exportar_csv(puntos_con_elev, slope_m_per_km, archivo_csv, author=args.author)
                    archivos_temporales.append(archivo_csv)

            if args.geojson or (not args.tramos and input(f"¿Exportar GeoJSON? (s/n) [n]: ").lower().strip() == 's'):
                with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                    archivo_geojson = tmp.name
                    exportar_geojson(puntos_con_elev, tramo_actual, archivo_geojson, author=args.author)
                    archivos_temporales.append(archivo_geojson)

            for archivo in archivos_temporales:
                if os.path.exists(archivo):
                    os.remove(archivo)
            print("-" * 20)

    except FileNotFoundError as e:
        logging.error(f"Error: Archivo no encontrado - {e}")
        print(f"ERROR: Archivo no encontrado - {e}")
    except ValueError as e:
        logging.error(f"Error en los datos o configuración: {e}")
        print(f"ERROR: Error en los datos o configuración - {e}")
    except Exception as e:
        logging.exception("Ocurrió un error inesperado:")
        print(f"\nERROR INESPERADO: {e}")

    logging.info(">>> Script finalizado <<<")

if __name__ == "__main__":
    main()
