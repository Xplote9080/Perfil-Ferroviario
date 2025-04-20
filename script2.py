# -*- coding: utf-8 -*-
"""
Script para generar perfiles altimétricos a partir de datos de estaciones para vías férreas.
© 2025 LAL - Todos los derechos reservados.

Funcionalidades:
- Carga estaciones desde un archivo CSV.
- Interpola puntos usando CubicSpline.
- Obtiene elevaciones con la API de Open-Meteo (con caché en CSV y llamadas por lotes paralelas).
- Calcula pendientes suavizadas (m/km) con filtro Savitzky-Golay.
- Genera gráficos HTML interactivos con Plotly (elevación + pendiente) con marca de agua.
- Exporta a PDF, KML, CSV y GeoJSON.
- Valida rangos de latitud, longitud y kilómetros.
- Incluye atribución: "Perfil altimétrico ferroviario - LAL 2025".
- Optimizado para Streamlit Cloud con progreso visual y manejo de archivos temporales.
"""

import csv
import requests
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.offline import plot
import simplekml
from geojson import Point, Feature, FeatureCollection, dump
import logging
import os
import tempfile
import pandas as pd
from typing import List, Optional, NamedTuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Configuración y Constantes ---
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CACHE_FILE = "elevations_cache.csv"
DEFAULT_INTERVAL_METERS = 200
REQUEST_TIMEOUT_SECONDS = 15
MAX_API_WORKERS = 4
MAX_BATCH_SIZE = 100
DEFAULT_ELEVATION_ON_ERROR = 0.0
DEFAULT_SMOOTH_WINDOW = 5
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

def load_cache_to_memory() -> Dict[tuple[float, float], float]:
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
        return
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
        
        if len(estaciones) < 2:
            raise ValueError("Se requieren al menos 2 estaciones")
        if not all(estaciones[i].km < estaciones[i+1].km for i in range(len(estaciones)-1)):
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
def _fetch_elevation_batch_from_api(batch_coords: List[tuple[float, float]], timeout: int = REQUEST_TIMEOUT_SECONDS) -> List[float]:
    """Obtiene elevaciones para un lote de coordenadas desde la API de Open-Meteo."""
    if not batch_coords:
        return []

    lats = [f"{c[0]:.5f}" for c in batch_coords]
    lons = [f"{c[1]:.5f}" for c in batch_coords]
    params = {'latitude': ','.join(lats), 'longitude': ','.join(lons)}

    try:
        response = requests.get(ELEVATION_API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        elevations = data.get("elevation", [])
        if not isinstance(elevations, list) or len(elevations) != len(batch_coords):
            raise ValueError(f"Expected {len(batch_coords)} elevations, got {len(elevations)}")
        return [float(e) if isinstance(e, (int, float)) else DEFAULT_ELEVATION_ON_ERROR for e in elevations]
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"Request failed for batch (first point: {lats[0]}, {lons[0]}): {e}")
        raise

def obtener_elevaciones_paralelo(puntos: List[InterpolatedPoint], author: str = AUTHOR_ATTRIBUTION, progress_callback: Optional[callable] = None) -> List[InterpolatedPoint]:
    """Obtiene elevaciones usando caché y API por lotes en paralelo."""
    if not puntos:
        return []

    logging.info(f"Iniciando consulta de elevaciones para {len(puntos)} puntos...")
    puntos_con_elevacion = [None] * len(puntos)
    points_to_fetch = []

    for i, punto in enumerate(puntos):
        cached_elev = _load_elevation_from_cache(punto.lat, punto.lon)
        if cached_elev is not None:
            puntos_con_elevacion[i] = punto._replace(elevation=cached_elev)
        else:
            points_to_fetch.append((i, punto))
        if progress_callback:
            progress_callback((i + 1) / (len(puntos) * 2))

    num_cached = len(puntos) - len(points_to_fetch)
    logging.info(f"{num_cached} puntos encontrados en caché.")

    if not points_to_fetch:
        return puntos_con_elevacion

    batches = []
    for i in range(0, len(points_to_fetch), MAX_BATCH_SIZE):
        batch = points_to_fetch[i:i + MAX_BATCH_SIZE]
        batch_coords = [(punto.lat, punto.lon) for _, punto in batch]
        batch_indices = [idx for idx, _ in batch]
        batches.append((batch_coords, batch_indices))

    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(_fetch_elevation_batch_from_api, coords): (coords, indices) for coords, indices in batches}
        for future in as_completed(futures):
            coords, indices = futures[future]
            try:
                elevations = future.result()
                for idx, elev in zip(indices, elevations):
                    punto = puntos[idx]
                    puntos_con_elevacion[idx] = punto._replace(elevation=elev)
                    _save_elevation_to_cache(punto.lat, punto.lon, elev, author)
            except Exception as e:
                logging.error(f"Error en lote: {e}")
                for idx in indices:
                    punto = puntos[idx]
                    puntos_con_elevacion[idx] = punto._replace(elevation=DEFAULT_ELEVATION_ON_ERROR)
            processed_count += len(indices)
            if progress_callback:
                progress_callback(0.5 + (processed_count / len(points_to_fetch)) / 2)

    return [p if p is not None else puntos[i]._replace(elevation=DEFAULT_ELEVATION_ON_ERROR) for i, p in enumerate(puntos_con_elevacion)]

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
        polyorder = min(2, window - 1)
        gradient = savgol_filter(valid_elevs, window, polyorder, deriv=1, delta=np.diff(valid_kms_m)[0])
        slope_m_per_km[valid_indices] = gradient * 1000.0
    except Exception as e:
        logging.error(f"Error al calcular pendiente suavizada: {e}")
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
    elev_color, slope_color = colors.split(',') if ',' in colors else ("blue", "orange")

    has_slope_data = slope_data is not None and len(slope_data) == len(kms)
    hover_texts = [
        f"<b>Km: {p.km:.3f}</b><br>Elev: {p.elevation:.1f} m<br>Pendiente: {slope_data[i]:.1f} m/km" if has_slope_data and not np.isnan(slope_data[i])
        else f"<b>Km: {p.km:.3f}</b><br>Elev: {p.elevation:.1f} m<br>Pendiente: N/A"
        for i, p in enumerate(puntos_con_elevacion)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=kms, y=elevs, mode='lines', name='Elevación',
        line=dict(color=elev_color, width=2),
        hoverinfo='text', text=hover_texts, yaxis='y1'
    ))

    if estaciones_tramo:
        station_kms = np.array([s.km for s in estaciones_tramo])
        indices_closest = np.searchsorted(kms, station_kms)
        indices_closest = np.clip(indices_closest, 0, len(kms) - 1)
        for i, est in enumerate(estaciones_tramo):
            idx = indices_closest[i]
            elev = elevs[idx] if not np.isnan(elevs[idx]) else DEFAULT_ELEVATION_ON_ERROR
            slope = slope_data[idx] if has_slope_data and not np.isnan(slope_data[idx]) else "N/A"
            fig.add_trace(go.Scatter(
                x=[kms[idx]], y=[elev],
                mode='markers+text', text=[est.nombre],
                textposition="top center",
                marker=dict(size=10, color='red', symbol='triangle-up'),
                name=est.nombre,
                hoverinfo='text',
                hovertext=f"<b>{est.nombre}</b><br>Km: {est.km:.3f}<br>Elev: {elev:.1f} m<br>Pendiente: {slope} m/km",
                yaxis='y1'
            ))

    if has_slope_data:
        fig.add_trace(go.Scattergl(
            x=kms, y=slope_data, mode='lines', name='Pendiente (m/km)',
            line=dict(color=slope_color, width=1.5, dash='dash'),
            yaxis='y2', hoverinfo='skip'
        ))

    template = "plotly" if theme == "light" else "plotly_dark"
    annotations = [
        dict(
            text=watermark,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=80, color="rgba(150, 150, 150, 0.3)"),
            textangle=-45,
            opacity=0.3
        )
    ] if watermark and watermark.lower() != "none" else []

    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center'),
        xaxis_title="Kilómetro",
        yaxis=dict(title="Elevación (msnm)", tickfont=dict(color=elev_color)),
        yaxis2=dict(
            title="Pendiente (m/km)", tickfont=dict(color=slope_color),
            anchor="x", overlaying="y", side="right", showgrid=False
        ),
        xaxis=dict(hoverformat='.3f'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=50),
        template=template,
        annotations=annotations
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
        fig.write_image(archivo_pdf, engine="kaleido")
        logging.info(f"PDF guardado en: {archivo_pdf}")
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
