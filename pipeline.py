from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from iduedu import get_closest_nodes
from pyproj import CRS
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=FutureWarning)

from blocksnet.analysis.diversity.shannon import shannon_diversity
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_spacematrix_morphotypes, get_strelka_morphotypes
from blocksnet.analysis.network.accessibility import area_accessibility
from blocksnet.analysis.provision import competitive_provision
from blocksnet.analysis.services.count import services_count
from blocksnet.analysis.services.density import services_density
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.relations import (
    accessibility_graph_to_gdfs,
    calculate_accessibility_matrix,
    validate_accessibility_graph,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INPUT_ROOT = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

WALK_SPEED_M_PER_MIN = 5_000 / 60
DEFAULT_WORK_CRS = 32635

LAND_USE_RULES = {
    "Индивидуальные жилые дома": LandUse.RESIDENTIAL,
    "Малоэтажные многоквартирные дома": LandUse.RESIDENTIAL,
    "Сренеэтажные жилые дома": LandUse.RESIDENTIAL,
    "Многоэтажные жилые дома": LandUse.RESIDENTIAL,
    "Зона исторической застройки": LandUse.RESIDENTIAL,
    "Многофункциональная общественно-деловая зона": LandUse.BUSINESS,
    "Специализированная общественная застройка": LandUse.SPECIAL,
    "Производственные зоны": LandUse.INDUSTRIAL,
    "Коммунально-складская зона": LandUse.INDUSTRIAL,
    "Транспортная инфраструктура": LandUse.TRANSPORT,
    "Инженерная инфраструктура": LandUse.SPECIAL,
    "Зона природных ландшафтов": LandUse.RECREATION,
    "Леса": LandUse.RECREATION,
    "Озеленённые территории общего пользования": LandUse.RECREATION,
    "Зоны отдыха": LandUse.RECREATION,
    "Акватории": LandUse.RECREATION,
    "Зоны с/х использования": LandUse.AGRICULTURE,
    "СНТ": LandUse.AGRICULTURE,
    "Зона ритуального назначения (кладбище)": LandUse.SPECIAL,
    "Режимные территории": LandUse.SPECIAL,
}

SERVICE_GROUPS = {
    "school": "образование",
    "kindergarten": "образование",
    "college": "образование",
    "university": "образование",
    "extra_education": "образование",
    "hospital": "здравоохранение",
    "polyclinic": "здравоохранение",
    "women_clinic": "здравоохранение",
    "dentist": "здравоохранение",
    "pharmacy": "здравоохранение",
    "rehabilitation": "здравоохранение",
    "emergency": "здравоохранение",
    "orphanage": "социальная сфера",
    "social_facility": "социальная сфера",
    "convenience": "торговля",
    "supermarket": "торговля",
    "mall": "торговля",
    "marketplace": "торговля",
    "market": "торговля",
    "clothes": "торговля",
    "electronic": "торговля",
    "household": "торговля",
    "houseware": "торговля",
    "bakery": "торговля",
    "cafe": "общепит",
    "restaurant": "общепит",
    "bar": "общепит",
    "bank": "услуги",
    "lawyer": "услуги",
    "notary": "услуги",
    "beauty": "услуги",
    "hairdresser": "услуги",
    "post_office": "услуги",
    "community_centre": "услуги",
    "mfz": "услуги",
    "library": "культура",
    "museum": "культура",
    "theatre": "культура",
    "cinema": "культура",
    "religion": "культура",
    "historic": "культура",
    "hotel": "туризм",
    "hostel": "туризм",
    "guest_house": "туризм",
    "bus_stop": "транспорт",
    "bus_station": "транспорт",
    "train_station": "транспорт",
    "train_building": "транспорт",
    "parking": "транспорт",
    "fuel": "транспорт",
    "pier": "транспорт",
    "park": "рекреация",
    "beach": "рекреация",
    "embankment": "рекреация",
    "pitch": "рекреация",
    "playground": "рекреация",
    "stadium": "рекреация",
    "swimming_pool": "рекреация",
    "sport_centre": "рекреация",
    "forest": "природа",
    "farmland": "природа",
    "substation": "инфраструктура",
    "water_works": "инфраструктура",
    "wastewater_plant": "инфраструктура",
    "boilerhouse": "инфраструктура",
    "pumping_station": "инфраструктура",
    "cell_tower": "инфраструктура",
    "engineering": "инфраструктура",
    "hydroelectric_pp": "инфраструктура",
    "cemetery": "специальное",
    "police": "специальное",
    "fire_station": "специальное",
    "outpost": "специальное",
    "recruitment": "специальное",
    "construction": "прочее",
    "chemical": "промышленность",
    "food_industry": "промышленность",
    "light_industry": "промышленность",
}

SOCIAL_SERVICE_GROUPS = (
    "образование",
    "здравоохранение",
    "социальная сфера",
    "культура",
    "рекреация",
)
CONSUMER_SERVICE_GROUPS = (
    "торговля",
    "общепит",
    "туризм",
    "услуги",
    "транспорт",
)
SERVICE_GROUP_SLUGS = {
    "образование": "education",
    "здравоохранение": "healthcare",
    "социальная сфера": "social_sphere",
    "культура": "culture",
    "рекреация": "recreation",
    "торговля": "retail",
    "общепит": "food",
    "туризм": "tourism",
    "услуги": "household_services",
    "транспорт": "transport",
    "инфраструктура": "infrastructure",
    "промышленность": "industry",
    "природа": "nature",
    "специальное": "special",
    "прочее": "other",
}
SERVICE_ACCESS_THRESHOLD_MIN = 15

PROVISION_COLUMNS = [
    "demand",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
    "provision_strong",
    "provision_weak",
]

BUILDING_COLUMNS = [
    "is_living",
    "number_of_floors",
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "population",
]


def resolve_paths() -> dict[str, Path]:
    candidates = sorted(INPUT_ROOT.rglob("boundary.geojson"))
    if not candidates:
        raise FileNotFoundError(f"Не найден boundary.geojson внутри {INPUT_ROOT}")
    territory_dir = next(
        (path.parent for path in candidates if (path.parent / "blocks.geojson").exists()),
        candidates[0].parent,
    )
    services_dir = next(
        (path.parent for path in territory_dir.rglob("school.geojson")),
        territory_dir / "сервисы",
    )
    for name in ("blocks.geojson", "building.geojson", "landuse.geojson", "roads.geojson"):
        path = territory_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Не найден обязательный слой: {path}")
    if not services_dir.exists():
        raise FileNotFoundError(f"Не найдена папка сервисов: {services_dir}")

    for folder in (PROCESSED_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "territory": territory_dir,
        "services": services_dir,
        "boundary": territory_dir / "boundary.geojson",
        "blocks": territory_dir / "blocks.geojson",
        "buildings": territory_dir / "building.geojson",
        "landuse": territory_dir / "landuse.geojson",
        "roads": territory_dir / "roads.geojson",
        "railways": territory_dir / "railways.geojson",
        "river": territory_dir / "river.geojson",
        "water": territory_dir / "water.geojson",
    }


def _clean_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    geom_name = out.geometry.name
    for column in out.columns:
        if column == geom_name:
            continue
        if isinstance(out[column].dtype, pd.CategoricalDtype):
            out[column] = out[column].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype(str)
        elif out[column].dtype == "object":
            out[column] = out[column].map(
                lambda value: value
                if value is None or isinstance(value, (str, int, float, bool))
                else str(value)
            )
    return out


def save_geojson(gdf: gpd.GeoDataFrame | None, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or len(gdf) == 0:
        print(f"Пустой слой, файл не создаю: {path.name}")
        return
    if gdf.crs is None:
        raise ValueError(f"У слоя {path.name} не задан CRS")
    out = gdf
    if out.index.name is not None and out.index.name in out.columns:
        out = out.reset_index(drop=True)
    _clean_for_geojson(out).to_crs(4326).to_file(path, driver="GeoJSON")
    print(f"Сохранено: {path}")


def _largest_polygon(geometry) -> Polygon | None:
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda part: part.area)
    if geometry.geom_type == "GeometryCollection":
        polygons = [part for part in geometry.geoms if part.geom_type == "Polygon"]
        if polygons:
            return max(polygons, key=lambda part: part.area)
    return None


def _iter_lines(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def load_inputs(paths: dict[str, Path]) -> dict[str, gpd.GeoDataFrame]:
    boundary_raw = gpd.read_file(paths["boundary"])
    if boundary_raw.crs is None:
        boundary_raw = boundary_raw.set_crs(4326)
    boundary_raw = boundary_raw.to_crs(4326)
    boundary_geom = boundary_raw.union_all().buffer(0)
    if boundary_geom.geom_type not in ("Polygon", "MultiPolygon"):
        boundary_geom = boundary_geom.convex_hull
    properties = boundary_raw.iloc[0].drop(labels=["geometry"]).to_dict()
    boundary_4326 = gpd.GeoDataFrame(
        [{**properties, "geometry": boundary_geom}],
        geometry="geometry",
        crs=4326,
    )

    blocks_raw = gpd.read_file(paths["blocks"])
    if blocks_raw.crs is None:
        blocks_raw = blocks_raw.set_crs(DEFAULT_WORK_CRS)
    work_crs = blocks_raw.crs.to_epsg() or DEFAULT_WORK_CRS

    layers = {
        "buildings": gpd.read_file(paths["buildings"]),
        "landuse": gpd.read_file(paths["landuse"]),
        "roads": gpd.read_file(paths["roads"]),
    }
    for name in ("railways", "river", "water"):
        layer_path = paths[name]
        layers[name] = gpd.read_file(layer_path) if layer_path.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
    for name, layer in layers.items():
        if layer.crs is None:
            layer = layer.set_crs(4326)
        layers[name] = layer.to_crs(work_crs)

    save_geojson(boundary_4326, PROCESSED_DIR / "boundary.geojson")
    save_geojson(blocks_raw, PROCESSED_DIR / "input_blocks.geojson")
    for name, layer in layers.items():
        save_geojson(layer, PROCESSED_DIR / f"{name}.geojson")

    return {
        "boundary_4326": boundary_4326,
        "blocks_raw": blocks_raw,
        "work_crs": int(work_crs),
        **layers,
    }


def prepare_blocks(blocks_raw: gpd.GeoDataFrame, work_crs: int) -> gpd.GeoDataFrame:
    blocks = blocks_raw.copy()
    if blocks.crs is None:
        blocks = blocks.set_crs(work_crs)
    blocks = blocks.to_crs(work_crs)
    blocks["source_fid"] = blocks["FID"] if "FID" in blocks.columns else pd.NA
    blocks["geometry"] = blocks.geometry.apply(_largest_polygon)
    blocks = blocks[blocks.geometry.notna() & ~blocks.geometry.is_empty].copy()
    blocks = blocks.drop(columns=["FID"], errors="ignore")
    blocks = blocks.reset_index(drop=True)
    blocks["block_id"] = np.arange(len(blocks), dtype=int)
    blocks = blocks.set_index("block_id", drop=False)
    blocks.index.name = "block_id"
    save_geojson(blocks, PROCESSED_DIR / "blocks.geojson")
    print(f"Кварталов: {len(blocks)}")
    print(f"Площадь кварталов: {blocks.area.sum() / 1_000_000:.2f} км²")
    return blocks


def prepare_land_use(blocks: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    zones = landuse.copy()
    if "function" in zones.columns and "functional_zone" not in zones.columns:
        zones = zones.rename(columns={"function": "functional_zone"})
    if "functional_zone" not in zones.columns:
        raise ValueError("В landuse.geojson нет колонки function/functional_zone")
    zones = zones[["functional_zone", "geometry"]].copy()
    zones = zones[zones["functional_zone"].notna()].copy()
    zones["functional_zone"] = zones["functional_zone"].astype(str)
    zones = zones.reset_index(drop=True)

    present = set(zones["functional_zone"].unique())
    unknown = sorted(present - set(LAND_USE_RULES))
    print("Зоны без правила:", unknown if unknown else "нет")

    blocks_land_use = assign_land_use(blocks, zones, LAND_USE_RULES)
    blocks_land_use["source_fid"] = blocks.loc[blocks_land_use.index, "source_fid"]
    blocks_land_use["block_id"] = blocks_land_use.index.astype(int)
    blocks_land_use = blocks_land_use.set_index("block_id", drop=False)
    blocks_land_use.index.name = "block_id"
    blocks_land_use["land_use"] = blocks_land_use["land_use"].apply(
        lambda value: value.value if hasattr(value, "value") else value
    )
    save_geojson(blocks_land_use, PROCESSED_DIR / "blocks_land_use.geojson")
    print("Распределение land use:")
    print(blocks_land_use["land_use"].value_counts(dropna=False).to_string())
    return blocks_land_use


def prepare_buildings(blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ready = buildings.copy()
    ready["is_living"] = ready["Is_living"].astype(bool) if "Is_living" in ready.columns else False
    if "number_of_floors" not in ready.columns and "building:levels" in ready.columns:
        ready["number_of_floors"] = ready["building:levels"]
    for column in BUILDING_COLUMNS:
        if column not in ready.columns:
            ready[column] = np.nan
        if column != "is_living":
            ready[column] = pd.to_numeric(ready[column], errors="coerce")

    missing_mask = ready[BUILDING_COLUMNS].isna().any(axis=1)
    if missing_mask.any():
        print(f"В зданиях есть пропуски ({int(missing_mask.sum())}), заполняю через impute_buildings")
        from blocksnet.preprocessing.imputing import impute_buildings

        ready = impute_buildings(
            ready,
            default_number_of_floors=1,
            living_area_coefficient=0.9,
            default_living_demand=30,
        )
    else:
        print("Пропусков в полях зданий нет — готовые атрибуты используются как есть.")

    ready = ready[BUILDING_COLUMNS + ["geometry"]].copy()
    buildings_blocks = aggregate_objects(blocks, ready)[0]
    buildings_blocks["block_id"] = buildings_blocks.index.astype(int)
    buildings_blocks = buildings_blocks.set_index("block_id", drop=False)
    buildings_blocks.index.name = "block_id"
    buildings_blocks["site_area"] = buildings_blocks.area
    buildings_blocks["count"] = pd.to_numeric(buildings_blocks["count"], errors="coerce").fillna(0)
    save_geojson(buildings_blocks, PROCESSED_DIR / "buildings_blocks.geojson")
    print(f"Зданий: {len(ready)}")
    print(f"Суммарное население: {buildings_blocks['population'].sum():,.0f} чел.")
    print(f"Площадь застройки: {buildings_blocks['footprint_area'].sum():,.0f} м²")
    return buildings_blocks


def service_sphere(service_group: str) -> str:
    if service_group in SOCIAL_SERVICE_GROUPS:
        return "social"
    if service_group in CONSUMER_SERVICE_GROUPS:
        return "consumer"
    return "other"


def _read_service_layer(path: Path, work_crs: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(work_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) == 0:
        return gdf
    service_type = path.stem
    name_column = next(
        (column for column in ("service_name", "srevice_name", "name") if column in gdf.columns),
        None,
    )
    gdf["service_type"] = service_type
    gdf["service_group"] = SERVICE_GROUPS.get(service_type, "прочее")
    gdf["service_label"] = gdf[name_column].fillna(service_type) if name_column else service_type
    if "capacity" in gdf.columns:
        gdf["capacity"] = pd.to_numeric(gdf["capacity"], errors="coerce")
    else:
        gdf["capacity"] = np.nan
    return gdf[["service_type", "service_group", "service_label", "capacity", "geometry"]].copy()


def prepare_services(
    blocks: gpd.GeoDataFrame, services_dir: Path, work_crs: int
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    service_layers = []
    for path in sorted(services_dir.glob("*.geojson")):
        layer = _read_service_layer(path, work_crs)
        if len(layer):
            service_layers.append(layer)
    if not service_layers:
        raise FileNotFoundError(f"Не найдено сервисных слоёв в {services_dir}")
    services_all = gpd.GeoDataFrame(
        pd.concat(service_layers, ignore_index=True),
        geometry="geometry",
        crs=work_crs,
    )
    services_all["service_sphere"] = services_all["service_group"].map(service_sphere)
    save_geojson(services_all, PROCESSED_DIR / "services_all.geojson")

    def aggregate_service(service_type: str, output_name: str, count_column: str, capacity_column: str):
        layer = services_all[services_all["service_type"] == service_type].copy()
        if len(layer) == 0:
            result = blocks[["geometry", "block_id"]].copy()
            result[capacity_column] = 0.0
            result[count_column] = 0
            result[f"{capacity_column}_is_estimate"] = False
            result.index = blocks.index
            return result
        capacity_was_missing = bool(layer["capacity"].isna().any())
        layer = layer[["capacity", "geometry"]].copy()
        if capacity_was_missing:
            from blocksnet.preprocessing.imputing import impute_services

            layer = impute_services(layer, service_type)
        result = aggregate_objects(blocks, layer)[0]
        result["block_id"] = result.index.astype(int)
        result = result.set_index("block_id", drop=False)
        result = result.rename(columns={"count": count_column, "capacity": capacity_column})
        result[f"{capacity_column}_is_estimate"] = capacity_was_missing
        save_geojson(result, PROCESSED_DIR / output_name)
        print(
            f"{service_type}: объектов {len(layer)}, ёмкость {result[capacity_column].sum():,.0f}, "
            f"оценка={capacity_was_missing}"
        )
        return result

    schools_blocks = aggregate_service(
        "school", "schools_blocks.geojson", "schools_count", "capacity"
    )
    kindergartens_blocks = aggregate_service(
        "kindergarten", "kindergartens_blocks.geojson", "kindergartens_count", "kindergarten_capacity"
    )

    schools_input = services_all[services_all["service_type"] == "school"][
        ["service_label", "capacity", "geometry"]
    ].rename(columns={"service_label": "name"})
    kindergartens_input = services_all[services_all["service_type"] == "kindergarten"][
        ["service_label", "capacity", "geometry"]
    ].rename(columns={"service_label": "name"})
    save_geojson(schools_input, PROCESSED_DIR / "schools.geojson")
    save_geojson(kindergartens_input, PROCESSED_DIR / "kindergartens.geojson")

    service_types = sorted(services_all["service_type"].unique())
    counts = services_all.copy()
    for service_type in service_types:
        counts[f"count_{service_type}"] = (counts["service_type"] == service_type).astype(int)
    counts = counts.drop(columns=["capacity"], errors="ignore")
    services_blocks = aggregate_objects(blocks, counts)[0]
    services_blocks["block_id"] = services_blocks.index.astype(int)
    services_blocks = services_blocks.set_index("block_id", drop=False)
    services_blocks["site_area"] = services_blocks.area

    count_df = services_count(services_blocks)
    density_df = services_density(services_blocks)
    diversity_df = shannon_diversity(services_blocks)

    services_metrics = pd.DataFrame(index=blocks.index)
    services_metrics.index.name = "block_id"
    services_metrics["services_count"] = count_df["count"]
    services_metrics["services_density"] = density_df["density"]
    services_metrics["shannon_diversity"] = diversity_df["shannon_diversity"]

    def sphere_service_metrics(sphere: str) -> pd.DataFrame:
        sphere_types = sorted(
            services_all.loc[services_all["service_sphere"] == sphere, "service_type"].unique()
        )
        result = pd.DataFrame(index=blocks.index)
        result.index.name = "block_id"
        if not sphere_types:
            result[f"services_{sphere}_count"] = 0.0
            result[f"services_{sphere}_density"] = 0.0
            result[f"services_{sphere}_diversity"] = 0.0
            return result

        sphere_objects = services_all[
            services_all["service_type"].isin(sphere_types)
        ][["geometry"]].copy()
        for service_type in sphere_types:
            sphere_objects[f"count_{service_type}"] = (
                services_all.loc[sphere_objects.index, "service_type"] == service_type
            ).astype(int)
        sphere_metrics = aggregate_objects(blocks, sphere_objects)[0]
        sphere_metrics["site_area"] = sphere_metrics.area
        sphere_count = services_count(sphere_metrics)["count"]
        sphere_density = services_density(sphere_metrics)["density"]
        sphere_diversity = shannon_diversity(sphere_metrics)["shannon_diversity"]
        result[f"services_{sphere}_count"] = sphere_count.reindex(blocks.index, fill_value=0)
        result[f"services_{sphere}_density"] = sphere_density.reindex(blocks.index, fill_value=0)
        result[f"services_{sphere}_diversity"] = sphere_diversity.reindex(
            blocks.index, fill_value=0
        )
        return result

    services_metrics = services_metrics.join(sphere_service_metrics("social"))
    services_metrics = services_metrics.join(sphere_service_metrics("consumer"))

    by_type = (
        services_all.groupby(
            ["service_type", "service_group", "service_sphere"], dropna=False
        )
        .agg(objects=("service_type", "size"), capacity_sum=("capacity", "sum"))
        .reset_index()
        .sort_values(["service_sphere", "service_group", "service_type"])
    )
    by_type.to_csv(RESULTS_DIR / "services_by_type.csv", index=False, encoding="utf-8")

    by_group = (
        services_all.groupby(["service_group", "service_sphere"], dropna=False)
        .agg(objects=("service_type", "size"), capacity_sum=("capacity", "sum"))
        .reset_index()
        .sort_values(["service_sphere", "service_group"])
    )
    by_group.to_csv(RESULTS_DIR / "services_by_group.csv", index=False, encoding="utf-8")

    print(f"Сервисных слоёв: {len(service_types)}, объектов: {len(services_all)}")
    print(f"Всего сервисов по кварталам: {services_metrics['services_count'].sum():,.0f}")
    print(
        "Социальные сервисы: "
        f"{services_metrics['services_social_count'].sum():,.0f}, "
        "коммерческие и бытовые: "
        f"{services_metrics['services_consumer_count'].sum():,.0f}"
    )

    return schools_blocks, kindergartens_blocks, services_metrics, by_type, services_all


def build_roads_graph(roads: gpd.GeoDataFrame, work_crs: int) -> nx.DiGraph:
    lines = []
    for geometry in roads.geometry:
        if geometry is None or geometry.is_empty:
            continue
        lines.append(shapely.set_precision(geometry, 0.5))
    if not lines:
        raise ValueError("В roads.geojson нет линейных объектов")
    merged = unary_union(lines)
    segments = [segment for segment in _iter_lines(merged) if segment.length > 0.01]
    if not segments:
        raise ValueError("Не удалось построить сегменты дорожной сети")

    graph = nx.DiGraph()
    node_ids: dict[tuple[float, float], int] = {}

    def get_node(x: float, y: float) -> int:
        key = (round(x, 3), round(y, 3))
        if key not in node_ids:
            node_id = len(node_ids)
            node_ids[key] = node_id
            graph.add_node(node_id, x=float(x), y=float(y))
        return node_ids[key]

    for segment in segments:
        coords = list(segment.coords)
        if len(coords) < 2:
            continue
        start = get_node(*coords[0][:2])
        end = get_node(*coords[-1][:2])
        if start == end:
            continue
        length = float(segment.length)
        time_min = length / WALK_SPEED_M_PER_MIN
        for source, target in ((start, end), (end, start)):
            if graph.has_edge(source, target):
                if time_min < graph[source][target]["time_min"]:
                    graph[source][target].update(length_meter=length, time_min=time_min, geometry=segment)
            else:
                graph.add_edge(
                    source,
                    target,
                    length_meter=length,
                    time_min=time_min,
                    geometry=segment,
                )

    graph.graph["crs"] = int(work_crs)
    graph.graph["walk_speed"] = WALK_SPEED_M_PER_MIN
    graph.graph["type"] = "walk"
    validate_accessibility_graph(graph)
    if graph.number_of_edges() == 0:
        raise RuntimeError("Дорожный граф пустой")
    print(f"Граф из roads.geojson: узлов {graph.number_of_nodes():,}, рёбер {graph.number_of_edges():,}")
    print(f"Компонент связности: {nx.number_weakly_connected_components(graph)}")
    return graph


def save_graph(graph: nx.DiGraph, prefix: str = "graph_walk") -> None:
    with open(PROCESSED_DIR / f"{prefix}.pickle", "wb") as file:
        pickle.dump(graph, file)
    nodes_gdf, edges_gdf = accessibility_graph_to_gdfs(graph)
    save_geojson(nodes_gdf, PROCESSED_DIR / f"{prefix}_nodes.geojson")
    save_geojson(edges_gdf, PROCESSED_DIR / f"{prefix}_edges.geojson")


def build_osm_graph(boundary_4326: gpd.GeoDataFrame) -> nx.Graph | None:
    try:
        import iduedu as ie

        print("Строю дополнительный пешеходный граф из OSM...")
        graph = ie.get_walk_graph(
            polygon=boundary_4326.union_all(),
            walk_speed=WALK_SPEED_M_PER_MIN,
        )
        if graph.number_of_edges() == 0:
            raise RuntimeError("OSM вернул пустой граф")
        crs_value = graph.graph.get("crs", 4326)
        if not isinstance(crs_value, (int, np.integer)):
            crs_value = CRS.from_user_input(crs_value).to_epsg()
        graph.graph["crs"] = int(crs_value)
        graph.graph["walk_speed"] = WALK_SPEED_M_PER_MIN
        graph.graph["type"] = "walk"
        validate_accessibility_graph(graph)
        save_graph(graph, "graph_walk_osm")
        print(f"OSM-граф: узлов {graph.number_of_nodes():,}, рёбер {graph.number_of_edges():,}")
        return graph
    except Exception as exc:
        print(f"OSM-граф не построен, продолжаю без него: {exc}")
        return None


def largest_connected_component_graph(graph: nx.Graph) -> nx.DiGraph:
    if not graph.is_directed():
        graph = graph.to_directed()
    components = sorted(nx.strongly_connected_components(graph), key=len)
    nodes = components[-1]
    subgraph = graph.subgraph(nodes).copy()
    mapping = {old: new for new, old in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, mapping)
    subgraph.graph.update(graph.graph)
    print(
        f"Для матрицы оставлен крупнейший компонент: "
        f"узлов {subgraph.number_of_nodes():,}, рёбер {subgraph.number_of_edges():,}"
    )
    return subgraph


def calculate_matrix(
    blocks: gpd.GeoDataFrame, graph: nx.Graph, output_name: str = "accessibility_matrix_walk_min.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_graph = largest_connected_component_graph(graph)
    acc_mx = calculate_accessibility_matrix(blocks, matrix_graph, weight_key="time_min", dtype=np.float32)
    acc_mx.index.name = "from_block_id"
    acc_mx.columns.name = "to_block_id"
    acc_mx.to_csv(RESULTS_DIR / output_name)

    values = acc_mx.to_numpy()
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    diagonal = np.diag(values)
    finite_diagonal = diagonal[np.isfinite(diagonal)]
    summary = pd.DataFrame(
        [
            {
                "Кварталов": len(blocks),
                "Пар всего": values.size,
                "Достижимых пар": int(finite_mask.sum()),
                "Недостижимых пар": int((~finite_mask).sum()),
                "Доля недостижимых": float((~finite_mask).mean()),
                "Минимум, мин": float(finite_values.min()) if finite_values.size else np.nan,
                "Медиана, мин": float(np.median(finite_values)) if finite_values.size else np.nan,
                "Максимум, мин": float(finite_values.max()) if finite_values.size else np.nan,
                "Максимум на диагонали, мин": float(np.abs(finite_diagonal).max())
                if finite_diagonal.size
                else np.nan,
            }
        ]
    )
    summary.to_csv(
        RESULTS_DIR / output_name.replace("accessibility_matrix_", "accessibility_summary_").replace(
            "_min.csv", ".csv"
        ),
        index=False,
        encoding="utf-8",
    )
    print("Матрица доступности:")
    print(summary.T.rename(columns={0: "Значение"}).to_string())
    return acc_mx, summary


def calculate_service_group_accessibility(
    blocks: gpd.GeoDataFrame,
    services_all: gpd.GeoDataFrame,
    graph: nx.Graph,
    threshold_min: int = SERVICE_ACCESS_THRESHOLD_MIN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    walk_graph = graph.to_undirected()
    block_nodes, _ = get_closest_nodes(blocks[["geometry"]], walk_graph)
    service_nodes, _ = get_closest_nodes(services_all[["geometry"]], walk_graph)
    services = services_all.assign(_node=service_nodes)

    reachable_by_node: dict[object, set[object]] = {}
    for node in set(block_nodes):
        reachable = nx.single_source_dijkstra_path_length(
            walk_graph, node, cutoff=threshold_min, weight="time_min"
        )
        reachable_by_node[node] = set(reachable)

    rows = []
    for service_group, group_services in services.groupby("service_group", sort=True):
        service_slug = SERVICE_GROUP_SLUGS.get(service_group, "other")
        sphere = service_sphere(service_group)
        group_nodes = sorted(set(group_services["_node"]))
        distances = (
            nx.multi_source_dijkstra_path_length(
                walk_graph, group_nodes, weight="time_min"
            )
            if group_nodes
            else {}
        )
        node_counts = group_services["_node"].value_counts()
        for block_id, block_node in zip(blocks.index, block_nodes):
            nearest_min = distances.get(block_node, np.nan)
            count_15min = int(
                sum(node_counts.get(node, 0) for node in reachable_by_node[block_node])
            )
            rows.append(
                {
                    "block_id": int(block_id),
                    "service_group": service_group,
                    "service_slug": service_slug,
                    "service_sphere": sphere,
                    "objects_total": len(group_services),
                    "nearest_min": float(nearest_min),
                    "count_15min": count_15min,
                }
            )

    access_long = pd.DataFrame(rows)
    access_long.to_csv(
        RESULTS_DIR / "service_group_accessibility.csv", index=False, encoding="utf-8"
    )

    nearest_wide = access_long.pivot(
        index="block_id", columns="service_slug", values="nearest_min"
    )
    nearest_wide.columns = [
        f"service_access_{column}_min" for column in nearest_wide.columns
    ]
    count_wide = access_long.pivot(
        index="block_id", columns="service_slug", values="count_15min"
    )
    count_wide.columns = [
        f"{column}_service_count_15min" for column in count_wide.columns
    ]
    access_wide = nearest_wide.join(count_wide).reindex(blocks.index)
    access_wide.index.name = "block_id"

    for sphere, groups in (
        ("social", SOCIAL_SERVICE_GROUPS),
        ("consumer", CONSUMER_SERVICE_GROUPS),
    ):
        slugs = [SERVICE_GROUP_SLUGS[group] for group in groups]
        nearest_columns = [
            f"service_access_{slug}_min"
            for slug in slugs
            if f"service_access_{slug}_min" in access_wide.columns
        ]
        count_columns = [
            f"{slug}_service_count_15min"
            for slug in slugs
            if f"{slug}_service_count_15min" in access_wide.columns
        ]
        access_wide[f"service_access_{sphere}_min"] = (
            access_wide[nearest_columns].min(axis=1, skipna=True)
            if nearest_columns
            else np.nan
        )
        access_wide[f"{sphere}_service_count_15min"] = (
            access_wide[count_columns].sum(axis=1) if count_columns else 0
        )

    all_nearest_columns = [
        f"service_access_{slug}_min"
        for slug in SERVICE_GROUP_SLUGS.values()
        if f"service_access_{slug}_min" in access_wide.columns
    ]
    all_count_columns = [
        f"{slug}_service_count_15min"
        for slug in SERVICE_GROUP_SLUGS.values()
        if f"{slug}_service_count_15min" in access_wide.columns
    ]
    access_wide["service_access_any_min"] = access_wide[all_nearest_columns].min(
        axis=1, skipna=True
    )
    access_wide["service_count_15min"] = access_wide[all_count_columns].sum(axis=1)
    access_wide.to_csv(
        RESULTS_DIR / "service_group_accessibility_wide.csv", encoding="utf-8"
    )

    summary = (
        access_long.groupby(["service_group", "service_sphere", "service_slug"], dropna=False)
        .agg(
            objects_total=("objects_total", "max"),
            median_nearest_min=("nearest_min", "median"),
            share_blocks_within_15=("count_15min", lambda values: float((values > 0).mean())),
            median_count_15=("count_15min", "median"),
        )
        .reset_index()
        .sort_values(["service_sphere", "median_nearest_min", "service_group"])
    )
    summary.to_csv(
        RESULTS_DIR / "service_group_summary.csv", index=False, encoding="utf-8"
    )

    print("Доступность сервисов по группам:")
    print(
        summary[
            ["service_group", "objects_total", "median_nearest_min", "share_blocks_within_15"]
        ].to_string(index=False)
    )
    return access_wide, summary, access_long


def _provision(
    buildings_blocks: gpd.GeoDataFrame,
    capacity_blocks: gpd.GeoDataFrame,
    capacity_column: str,
    service_type: str,
    acc_mx: pd.DataFrame,
    accessible_column: str,
    csv_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = service_types_config[service_type]
    provision_input = buildings_blocks[["population"]].join(capacity_blocks[[capacity_column]])
    provision_input = provision_input.rename(columns={capacity_column: "capacity"})
    provision_input["capacity"] = provision_input["capacity"].fillna(0).round().astype(int)

    if provision_input["population"].sum() > 0 and provision_input["capacity"].sum() > 0:
        provision_df, links = competitive_provision(
            provision_input,
            acc_mx,
            int(config["accessibility"]),
            int(config["demand"]),
        )
    else:
        print(f"Provision для {service_type} пропущен: нет населения или ёмкости")
        provision_df = pd.DataFrame(index=provision_input.index)
        for column in PROVISION_COLUMNS:
            provision_df[column] = 0.0
        provision_df["demand"] = 0
        links = pd.DataFrame(columns=["source", "target", "value"])

    provision_df.index.name = "block_id"
    provision_df.to_csv(RESULTS_DIR / csv_name, encoding="utf-8")
    links.to_csv(RESULTS_DIR / csv_name.replace(".csv", "_links.csv"), encoding="utf-8")
    print(
        f"{service_type}: доступность {int(config['accessibility'])} мин, "
        f"норматив {int(config['demand'])} на 1000 жителей, "
        f"сильная обеспеченность {provision_df[accessible_column].mean():.3f}"
    )
    return provision_df, links


def compute_metrics(
    blocks: gpd.GeoDataFrame,
    blocks_land_use: gpd.GeoDataFrame,
    buildings_blocks: gpd.GeoDataFrame,
    schools_blocks: gpd.GeoDataFrame,
    kindergartens_blocks: gpd.GeoDataFrame,
    services_metrics: pd.DataFrame,
    service_access: pd.DataFrame,
    acc_mx: pd.DataFrame,
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    buildings_blocks = buildings_blocks.copy()
    buildings_blocks["site_area"] = buildings_blocks.area
    density_df = calculate_density_indicators(buildings_blocks)
    density_df.index.name = "block_id"
    density_df.to_csv(RESULTS_DIR / "density.csv", encoding="utf-8")

    developed = density_df[density_df["fsi"] > 0]
    if len(developed) >= 2:
        n_clusters = min(11, len(developed))
        spacematrix_df, spacematrix_clusters = get_spacematrix_morphotypes(
            density_df,
            n_clusters=n_clusters,
            random_state=42,
        )
        spacematrix_df.index.name = "block_id"
        spacematrix_clusters.index.name = "cluster_id"
        spacematrix_df.to_csv(RESULTS_DIR / "spacematrix.csv", encoding="utf-8")
        spacematrix_clusters.to_csv(RESULTS_DIR / "spacematrix_clusters.csv", encoding="utf-8")
    else:
        print("Spacematrix пропущен: слишком мало кварталов с застройкой")
        spacematrix_df = pd.DataFrame(index=density_df.index)
        spacematrix_df["cluster"] = pd.NA
        spacematrix_df["morphotype"] = pd.NA
        spacematrix_clusters = pd.DataFrame()

    strelka_df = get_strelka_morphotypes(density_df)
    strelka_df.index.name = "block_id"
    strelka_df.to_csv(RESULTS_DIR / "strelka.csv", encoding="utf-8")

    area_acc = area_accessibility(acc_mx, density_df[["site_area"]])
    area_acc.index.name = "block_id"
    area_acc.to_csv(RESULTS_DIR / "area_accessibility.csv", encoding="utf-8")

    provision_school, school_links = _provision(
        buildings_blocks,
        schools_blocks,
        "capacity",
        "school",
        acc_mx,
        "provision_strong",
        "provision_school.csv",
    )
    provision_kindergarten, kindergarten_links = _provision(
        buildings_blocks,
        kindergartens_blocks,
        "kindergarten_capacity",
        "kindergarten",
        acc_mx,
        "provision_strong",
        "provision_kindergarten.csv",
    )

    land_use_columns = [land_use.value for land_use in LandUse]
    base_columns = ["geometry", "land_use", "share", "source_fid"] + [
        column for column in land_use_columns if column in blocks_land_use.columns
    ]
    base_columns = [column for column in base_columns if column in blocks_land_use.columns]
    metrics = blocks_land_use[base_columns].copy()

    building_columns = [column for column in BUILDING_COLUMNS + ["count"] if column in buildings_blocks.columns]
    buildings_part = buildings_blocks[building_columns].rename(columns={"count": "buildings_count"})
    metrics = metrics.join(buildings_part, how="left")

    school_columns = [
        column
        for column in ["capacity", "schools_count", "capacity_is_estimate"]
        if column in schools_blocks.columns
    ]
    metrics = metrics.join(schools_blocks[school_columns], how="left")

    kindergarten_columns = [
        column
        for column in ["kindergarten_capacity", "kindergartens_count", "kindergarten_capacity_is_estimate"]
        if column in kindergartens_blocks.columns
    ]
    metrics = metrics.join(kindergartens_blocks[kindergarten_columns], how="left")

    density_columns = [
        column
        for column in ["site_area", "fsi", "gsi", "mxi", "l", "osr", "share_living", "share_non_living"]
        if column in density_df.columns
    ]
    metrics = metrics.join(density_df[density_columns], how="left")

    spacematrix_columns = [column for column in ["cluster", "morphotype"] if column in spacematrix_df.columns]
    if spacematrix_columns:
        metrics = metrics.join(
            spacematrix_df[spacematrix_columns].rename(
                columns={"cluster": "spacematrix_cluster", "morphotype": "spacematrix_morphotype"}
            ),
            how="left",
        )
    else:
        metrics["spacematrix_cluster"] = pd.NA
        metrics["spacematrix_morphotype"] = pd.NA

    if "morphotype" in strelka_df.columns:
        metrics = metrics.join(
            strelka_df[["morphotype"]].rename(columns={"morphotype": "strelka_morphotype"}),
            how="left",
        )
    else:
        metrics["strelka_morphotype"] = pd.NA

    provision_columns = [column for column in PROVISION_COLUMNS if column in provision_school.columns]
    metrics = metrics.join(provision_school[provision_columns], how="left")
    kindergarten_provision = provision_kindergarten[provision_columns].rename(
        columns={column: f"kindergarten_{column}" for column in provision_columns}
    )
    metrics = metrics.join(kindergarten_provision, how="left")
    metrics = metrics.join(area_acc, how="left")
    metrics = metrics.join(services_metrics, how="left")
    metrics = metrics.join(service_access, how="left")

    metrics["block_id"] = metrics.index.astype(int)
    metrics.index.name = "block_id"

    metrics_for_csv = metrics.drop(columns=["block_id"], errors="ignore").copy()
    metrics_for_csv.to_csv(RESULTS_DIR / "blocks_metrics.csv", encoding="utf-8")

    metrics_for_geojson = metrics.drop(columns=["block_id"], errors="ignore").copy()
    metrics_for_geojson.index.name = "block_id"
    for column in metrics_for_geojson.select_dtypes(include=[np.number]).columns:
        if np.isinf(metrics_for_geojson[column]).any():
            metrics_for_geojson[f"{column}_is_inf"] = np.isinf(metrics_for_geojson[column])
            metrics_for_geojson[column] = metrics_for_geojson[column].replace([np.inf, -np.inf], np.nan)
    save_geojson(metrics_for_geojson, RESULTS_DIR / "blocks_metrics.geojson")

    print(f"Строк в итоговом слое: {len(metrics)}")
    print("Колонки итогового слоя:")
    print(", ".join(metrics.columns))
    return {
        "density": density_df,
        "spacematrix": spacematrix_df,
        "spacematrix_clusters": spacematrix_clusters,
        "strelka": strelka_df,
        "provision_school": provision_school,
        "provision_kindergarten": provision_kindergarten,
        "area_accessibility": area_acc,
        "service_access": service_access,
        "metrics": metrics,
    }


def validate_outputs(
    blocks: gpd.GeoDataFrame,
    acc_mx: pd.DataFrame,
    metrics: pd.DataFrame,
    schools_blocks: gpd.GeoDataFrame,
    kindergartens_blocks: gpd.GeoDataFrame,
) -> pd.DataFrame:
    checks = []

    def check(name: str, condition: bool, details: str = "") -> None:
        condition = bool(condition)
        checks.append({"Проверка": name, "OK": condition, "Подробности": details})
        print(("OK  " if condition else "FAIL"), name, details)

    check("Кварталы созданы", len(blocks) > 0, f"{len(blocks)} кварталов")
    check("Индексы кварталов уникальны", blocks.index.is_unique)
    check("Матрица квадратная", acc_mx.shape == (len(blocks), len(blocks)), f"{acc_mx.shape}")
    values = acc_mx.to_numpy()
    finite_values = values[np.isfinite(values)]
    check(
        "Нет отрицательных времён",
        finite_values.size == 0 or finite_values.min() >= 0,
        f"минимум {finite_values.min():.3f}" if finite_values.size else "нет данных",
    )
    diagonal = np.diag(values)
    finite_diagonal = diagonal[np.isfinite(diagonal)]
    check(
        "Диагональ около нуля",
        finite_diagonal.size > 0 and np.abs(finite_diagonal).max() < 1e-4,
        f"максимум {np.abs(finite_diagonal).max():.6f}" if finite_diagonal.size else "нет данных",
    )

    population = pd.to_numeric(metrics.get("population"), errors="coerce").fillna(0)
    check("Население не отрицательное", (population >= 0).all(), f"сумма {population.sum():,.0f} чел.")
    gsi = pd.to_numeric(metrics.get("gsi"), errors="coerce").dropna()
    check("GSI не больше 1", gsi.empty or (gsi <= 1 + 1e-6).all(), f"максимум {gsi.max():.3f}" if not gsi.empty else "нет данных")
    provision = pd.to_numeric(metrics.get("provision_strong"), errors="coerce").dropna()
    check(
        "Обеспеченность школами в диапазоне 0–1",
        provision.empty or ((provision >= -1e-6) & (provision <= 1 + 1e-6)).all(),
        f"от {provision.min():.3f} до {provision.max():.3f}" if not provision.empty else "нет данных",
    )
    kindergarten_provision = pd.to_numeric(metrics.get("kindergarten_provision_strong"), errors="coerce").dropna()
    check(
        "Обеспеченность детсадами в диапазоне 0–1",
        kindergarten_provision.empty
        or ((kindergarten_provision >= -1e-6) & (kindergarten_provision <= 1 + 1e-6)).all(),
        f"от {kindergarten_provision.min():.3f} до {kindergarten_provision.max():.3f}"
        if not kindergarten_provision.empty
        else "нет данных",
    )
    check(
        "Вместимость школ из исходных данных",
        "capacity_is_estimate" in metrics.columns and not bool(metrics["capacity_is_estimate"].fillna(False).any()),
    )
    check(
        "Вместимость детсадов из исходных данных",
        "kindergarten_capacity_is_estimate" in metrics.columns
        and not bool(metrics["kindergarten_capacity_is_estimate"].fillna(False).any()),
    )
    check(
        "Школы агрегированы",
        "schools_count" in metrics.columns and pd.to_numeric(metrics["schools_count"], errors="coerce").fillna(0).sum() > 0,
        f"{pd.to_numeric(metrics.get('schools_count'), errors='coerce').fillna(0).sum():.0f} школ",
    )
    check(
        "Детские сады агрегированы",
        "kindergartens_count" in metrics.columns
        and pd.to_numeric(metrics["kindergartens_count"], errors="coerce").fillna(0).sum() > 0,
        f"{pd.to_numeric(metrics.get('kindergartens_count'), errors='coerce').fillna(0).sum():.0f} садов",
    )
    check(
        "Сервисы посчитаны",
        "services_count" in metrics.columns and pd.to_numeric(metrics["services_count"], errors="coerce").fillna(0).sum() > 0,
        f"{pd.to_numeric(metrics.get('services_count'), errors='coerce').fillna(0).sum():.0f} объектов",
    )
    for sphere in ("social", "consumer"):
        count_column = f"services_{sphere}_count"
        access_column = f"service_access_{sphere}_min"
        check(
            f"Сервисы сферы «{sphere}» посчитаны",
            count_column in metrics.columns
            and pd.to_numeric(metrics[count_column], errors="coerce").fillna(0).sum() > 0,
            f"{pd.to_numeric(metrics.get(count_column), errors='coerce').fillna(0).sum():.0f} объектов",
        )
        access_values = pd.to_numeric(metrics.get(access_column), errors="coerce").dropna()
        check(
            f"Доступность сферы «{sphere}» рассчитана",
            access_column in metrics.columns
            and len(access_values) > 0
            and (access_values >= 0).all(),
            f"медиана {access_values.median():.1f} мин"
            if len(access_values)
            else "нет данных",
        )
    group_access_columns = [
        column
        for column in metrics.columns
        if column.startswith("service_access_")
        and column.endswith("_min")
        and column not in {"service_access_social_min", "service_access_consumer_min", "service_access_any_min"}
    ]
    check(
        "Доступность по группам сервисов рассчитана",
        len(group_access_columns) >= len(SOCIAL_SERVICE_GROUPS) + len(CONSUMER_SERVICE_GROUPS),
        f"{len(group_access_columns)} групп",
    )
    land_use_missing = int(metrics["land_use"].isna().sum()) if "land_use" in metrics.columns else len(metrics)
    check("Land use назначен большинству кварталов", land_use_missing < len(metrics) / 2, f"без land use: {land_use_missing}")

    expected_files = [
        PROCESSED_DIR / "boundary.geojson",
        PROCESSED_DIR / "blocks.geojson",
        PROCESSED_DIR / "blocks_land_use.geojson",
        PROCESSED_DIR / "buildings_blocks.geojson",
        PROCESSED_DIR / "schools_blocks.geojson",
        PROCESSED_DIR / "kindergartens_blocks.geojson",
        PROCESSED_DIR / "services_all.geojson",
        PROCESSED_DIR / "graph_walk.pickle",
        RESULTS_DIR / "accessibility_matrix_walk_min.csv",
        RESULTS_DIR / "accessibility_summary_walk.csv",
        RESULTS_DIR / "service_group_accessibility.csv",
        RESULTS_DIR / "service_group_accessibility_wide.csv",
        RESULTS_DIR / "service_group_summary.csv",
        RESULTS_DIR / "services_by_group.csv",
        RESULTS_DIR / "blocks_metrics.geojson",
        RESULTS_DIR / "blocks_metrics.csv",
    ]
    for path in expected_files:
        check(f"Файл существует: {path.name}", path.exists(), str(path))

    for path in (
        PROCESSED_DIR / "boundary.geojson",
        PROCESSED_DIR / "blocks.geojson",
        PROCESSED_DIR / "blocks_land_use.geojson",
        RESULTS_DIR / "blocks_metrics.geojson",
    ):
        if path.exists():
            layer = gpd.read_file(path)
            check(
                f"CRS EPSG:4326: {path.name}",
                layer.crs is not None and layer.crs.to_epsg() == 4326,
                str(layer.crs),
            )

    validation_df = pd.DataFrame(checks)
    validation_df.to_csv(RESULTS_DIR / "validation_summary.csv", index=False, encoding="utf-8")
    with open(RESULTS_DIR / "validation_summary.txt", "w", encoding="utf-8") as file:
        file.write(validation_df.to_string(index=False))
    return validation_df


def run_pipeline(build_osm: bool = False) -> dict[str, object]:
    paths = resolve_paths()
    print(f"Территория: {paths['territory']}")
    print(f"Сервисы: {paths['services']}")

    inputs = load_inputs(paths)
    blocks = prepare_blocks(inputs["blocks_raw"], inputs["work_crs"])
    blocks_land_use = prepare_land_use(blocks, inputs["landuse"])
    buildings_blocks = prepare_buildings(blocks, inputs["buildings"])
    schools_blocks, kindergartens_blocks, services_metrics, services_by_type, services_all = prepare_services(
        blocks, paths["services"], inputs["work_crs"]
    )

    graph = build_roads_graph(inputs["roads"], inputs["work_crs"])
    save_graph(graph, "graph_walk")
    acc_mx, accessibility_summary = calculate_matrix(blocks, graph)
    service_access, service_access_summary, service_access_long = (
        calculate_service_group_accessibility(blocks, services_all, graph)
    )

    if build_osm:
        osm_graph = build_osm_graph(inputs["boundary_4326"])
        if osm_graph is not None:
            osm_mx, osm_summary = calculate_matrix(
                blocks, osm_graph, output_name="accessibility_matrix_walk_osm_min.csv"
            )
            comparison = pd.DataFrame(
                [
                    {"Вариант": "roads.geojson", "Медиана, мин": float(accessibility_summary["Медиана, мин"].iloc[0]), "Недостижимых пар": int(accessibility_summary["Недостижимых пар"].iloc[0])},
                    {"Вариант": "OSM", "Медиана, мин": float(osm_summary["Медиана, мин"].iloc[0]), "Недостижимых пар": int(osm_summary["Недостижимых пар"].iloc[0])},
                ]
            )
            comparison.to_csv(RESULTS_DIR / "accessibility_comparison.csv", index=False, encoding="utf-8")
            print("Сравнение графов:")
            print(comparison.to_string(index=False))
    else:
        osm_graph = None

    computed = compute_metrics(
        blocks,
        blocks_land_use,
        buildings_blocks,
        schools_blocks,
        kindergartens_blocks,
        services_metrics,
        service_access,
        acc_mx,
    )
    validation = validate_outputs(blocks, acc_mx, computed["metrics"], schools_blocks, kindergartens_blocks)

    metrics = computed["metrics"]
    total_population = float(pd.to_numeric(metrics["population"], errors="coerce").fillna(0).sum())
    school_capacity = float(pd.to_numeric(metrics["capacity"], errors="coerce").fillna(0).sum())
    kindergarten_capacity = float(
        pd.to_numeric(metrics["kindergarten_capacity"], errors="coerce").fillna(0).sum()
    )
    print("\nКраткая сводка")
    print(f"Кварталов: {len(blocks)}")
    print(f"Население: {total_population:,.0f} чел.")
    print(f"Вместимость школ: {school_capacity:,.0f} мест")
    print(f"Вместимость детсадов: {kindergarten_capacity:,.0f} мест")
    print(f"Сервисов: {pd.to_numeric(metrics['services_count'], errors='coerce').fillna(0).sum():,.0f}")
    print(f"Медиана времени в пути: {float(accessibility_summary['Медиана, мин'].iloc[0]):.1f} мин")
    print(f"Проверок выполнено: {len(validation)}, провалено: {int((~validation['OK']).sum())}")

    return {
        "paths": paths,
        "blocks": blocks,
        "blocks_land_use": blocks_land_use,
        "buildings_blocks": buildings_blocks,
        "schools_blocks": schools_blocks,
        "kindergartens_blocks": kindergartens_blocks,
        "services_metrics": services_metrics,
        "services_by_type": services_by_type,
        "services_all": services_all,
        "service_access": service_access,
        "service_access_summary": service_access_summary,
        "service_access_long": service_access_long,
        "graph": graph,
        "osm_graph": osm_graph,
        "acc_mx": acc_mx,
        "accessibility_summary": accessibility_summary,
        "validation": validation,
        **computed,
    }


def main() -> None:
    build_osm = "--osm" in sys.argv
    run_pipeline(build_osm=build_osm)
    if "--report" in sys.argv:
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "report.py")], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
