from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = ROOT / "dashboard.html"


def read_csv_index(path):
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "block_id" in frame.columns:
        frame = frame.set_index("block_id")
    return frame


def with_block_index(frame):
    frame = frame.copy()
    if "block_id" in frame.columns:
        frame["block_id"] = pd.to_numeric(frame["block_id"], errors="coerce")
        frame = frame.set_index("block_id")
    frame.index.name = "block_id"
    return frame


def load_metrics():
    base = gpd.read_file(RESULTS_DIR / "blocks_metrics.geojson")
    base = with_block_index(base)

    density = read_csv_index(RESULTS_DIR / "density.csv")
    spacematrix = read_csv_index(RESULTS_DIR / "spacematrix.csv")
    provision = read_csv_index(RESULTS_DIR / "provision_school.csv")
    buildings = with_block_index(gpd.read_file(PROCESSED_DIR / "buildings_blocks.geojson"))
    schools = with_block_index(gpd.read_file(PROCESSED_DIR / "schools_blocks.geojson"))

    for column in ["site_area"]:
        if column in density.columns:
            base = base.drop(columns=[column], errors="ignore")
    density_columns = [column for column in ["site_area"] if column in density.columns]
    if density_columns:
        base = base.join(density[density_columns], how="left")

    if "cluster" in spacematrix.columns:
        base = base.drop(columns=["spacematrix_cluster"], errors="ignore")
        base = base.join(
            spacematrix[["cluster"]].rename(columns={"cluster": "spacematrix_cluster"}),
            how="left",
        )

    provision_columns = [
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
    provision_columns = [
        column for column in provision_columns if column in provision.columns
    ]
    for column in provision_columns:
        base = base.drop(columns=[column], errors="ignore")
    if provision_columns:
        base = base.join(provision[provision_columns], how="left")

    building_columns = [
        column
        for column in ["is_living", "number_of_floors", "count"]
        if column in buildings.columns
    ]
    buildings_part = buildings[building_columns].rename(
        columns={"count": "buildings_count"}
    )
    for column in buildings_part.columns:
        base = base.drop(columns=[column], errors="ignore")
    if len(buildings_part.columns):
        base = base.join(buildings_part, how="left")

    school_columns = [column for column in ["count"] if column in schools.columns]
    schools_part = schools[school_columns].rename(columns={"count": "schools_count"})
    for column in schools_part.columns:
        base = base.drop(columns=[column], errors="ignore")
    if len(schools_part.columns):
        base = base.join(schools_part, how="left")

    if "capacity_is_estimate" not in base.columns:
        base["capacity_is_estimate"] = False
    if "kindergarten_capacity_is_estimate" not in base.columns:
        base["kindergarten_capacity_is_estimate"] = False
    base["capacity_is_estimate"] = base["capacity_is_estimate"].fillna(False).astype(bool)
    base["kindergarten_capacity_is_estimate"] = (
        base["kindergarten_capacity_is_estimate"].fillna(False).astype(bool)
    )

    base = base.copy()
    base["block_id"] = base.index
    return base


def shorten_paths(text):
    root_text = str(ROOT)
    variants = {
        root_text,
        root_text.replace("\\", "/"),
        root_text.replace("/", "\\"),
    }
    for variant in variants:
        text = text.replace(variant, "...\\" + ROOT.name)
    return text


def fmt(value, digits=2):
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "да" if bool(value) else "нет"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        return f"{float(value):,.{digits}f}".replace(",", " ")
    return html.escape(shorten_paths(str(value)))


def render_table(frame, columns=None, labels=None, table_id=None):
    if frame is None or len(frame) == 0:
        return "<p>Нет данных.</p>"
    if columns is None:
        columns = list(frame.columns)
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return "<p>Нет данных.</p>"
    data = {}
    for column in columns:
        label = labels.get(column, column) if labels else column
        data[label] = frame[column].map(fmt)
    view = pd.DataFrame(data)
    result = view.to_html(index=False, classes="data-table", border=0, escape=True)
    if table_id:
        result = result.replace("<table ", f'<table id="{table_id}" ', 1)
    return result


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def chart_card(title, image_base64, note=""):
    return (
        '<figure class="chart">'
        f'<img src="data:image/png;base64,{image_base64}" alt="{html.escape(title)}">'
        f'<figcaption><b>{html.escape(title)}</b>. {html.escape(note)}</figcaption>'
        "</figure>"
    )


def make_charts(frame):
    charts = []

    area = pd.to_numeric(frame.get("area_accessibility"), errors="coerce").dropna()
    if len(area):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(area, bins=20, color="#4c78a8", edgecolor="white")
        ax.set_xlabel("минуты")
        ax.set_ylabel("кварталов")
        ax.set_title("Зональная доступность")
        charts.append(chart_card("Зональная доступность", fig_to_base64(fig), "Чем меньше, тем ближе квартал ко всей территории."))

    land = frame.copy()
    land["land_use"] = land["land_use"].fillna("не назначено")
    if "population" in land.columns:
        summary = (
            land.groupby("land_use", dropna=False)
            .agg(population=("population", "sum"))
            .sort_values("population")
        )
        summary = summary[summary["population"] > 0]
        if len(summary):
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(summary.index, summary["population"], color="#f58518")
            ax.set_xlabel("человек")
            ax.set_title("Население по землепользованию")
            charts.append(chart_card("Население по землепользованию", fig_to_base64(fig), "Сумма атрибута population зданий в кварталах."))

    if {"demand", "demand_within"}.issubset(frame.columns):
        total = pd.to_numeric(frame["demand"], errors="coerce").fillna(0).sum()
        within = pd.to_numeric(frame["demand_within"], errors="coerce").fillna(0).sum()
        outside = max(total - within, 0)
        if total > 0:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(["В 15 мин", "Не покрыто"], [within, outside], color=["#54a24b", "#e45756"])
            ax.set_ylabel("мест")
            ax.set_title("Школьный спрос")
            charts.append(chart_card("Школьный спрос", fig_to_base64(fig), "Сколько мест спроса покрыто в пределах 15 минут."))

    if {"kindergarten_demand", "kindergarten_demand_within"}.issubset(frame.columns):
        total = pd.to_numeric(frame["kindergarten_demand"], errors="coerce").fillna(0).sum()
        within = pd.to_numeric(frame["kindergarten_demand_within"], errors="coerce").fillna(0).sum()
        outside = max(total - within, 0)
        if total > 0:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(["В 7 мин", "Не покрыто"], [within, outside], color=["#54a24b", "#e45756"])
            ax.set_ylabel("мест")
            ax.set_title("Спрос на детские сады")
            charts.append(chart_card("Спрос на детские сады", fig_to_base64(fig), "Сколько мест спроса покрыто в пределах 7 минут."))

    if "services_count" in frame.columns:
        services = pd.to_numeric(frame["services_count"], errors="coerce").fillna(0)
        if services.sum() > 0:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(services, bins=15, color="#72b7b2", edgecolor="white")
            ax.set_xlabel("объектов на квартал")
            ax.set_ylabel("кварталов")
            ax.set_title("Распределение сервисов")
            charts.append(chart_card("Распределение сервисов", fig_to_base64(fig), "Число объектов обслуживания в границах квартала."))

    if "shannon_diversity" in frame.columns:
        diversity = pd.to_numeric(frame["shannon_diversity"], errors="coerce").dropna()
        if len(diversity):
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(diversity, bins=15, color="#b279a2", edgecolor="white")
            ax.set_xlabel("индекс Шеннона")
            ax.set_ylabel("кварталов")
            ax.set_title("Разнообразие сервисов")
            charts.append(chart_card("Разнообразие сервисов", fig_to_base64(fig), "Индекс Шеннона по типам объектов обслуживания."))

    if "spacematrix_morphotype" in frame.columns:
        morph = (
            frame["spacematrix_morphotype"]
            .fillna("нет данных")
            .value_counts()
            .head(8)
            .sort_values()
        )
        if len(morph):
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(morph.index, morph.values, color="#72b7b2")
            ax.set_xlabel("кварталов")
            ax.set_title("Морфотипы Spacematrix")
            charts.append(chart_card("Морфотипы Spacematrix", fig_to_base64(fig), "Модельные классы BlocksNet, а не официальная типология."))

    if {"fsi", "gsi"}.issubset(frame.columns):
        plot_data = frame[["fsi", "gsi"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(plot_data):
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.scatter(plot_data["fsi"], plot_data["gsi"], s=22, alpha=0.7, color="#e45756")
            ax.set_xlabel("FSI")
            ax.set_ylabel("GSI")
            ax.set_title("Плотность застройки")
            charts.append(chart_card("Плотность застройки", fig_to_base64(fig), "FSI — интенсивность, GSI — застроенность участка."))

    group_summary_path = RESULTS_DIR / "service_group_summary.csv"
    if group_summary_path.exists():
        group_summary = pd.read_csv(group_summary_path)
        group_summary = group_summary[
            group_summary["service_sphere"].isin(["social", "consumer"])
        ].copy()
        if len(group_summary):
            group_summary["label"] = group_summary["service_group"]
            group_summary = group_summary.sort_values("median_nearest_min")
            colors = [
                "#2f6f9f" if sphere == "social" else "#f58518"
                for sphere in group_summary["service_sphere"]
            ]
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.barh(group_summary["label"], group_summary["median_nearest_min"], color=colors)
            ax.set_xlabel("минуты")
            ax.set_title("Медианная доступность по группам")
            charts.append(
                chart_card(
                    "Доступность сервисов по группам",
                    fig_to_base64(fig),
                    "Медиана времени до ближайшего объекта группы по кварталам; синий — социальная сфера, оранжевый — коммерческая.",
                )
            )

    return "\n".join(charts)


def gradient_legend(colors, minimum, maximum, label):
    gradient = ",".join(colors)
    return (
        '<div class="legend">'
        f'<div class="legend-title">{html.escape(label)}</div>'
        f'<div class="gradient" style="background: linear-gradient(to right, {gradient});"></div>'
        f'<div class="legend-range"><span>{fmt(minimum)}</span><span>{fmt(maximum)}</span></div>'
        "</div>"
    )


def categorical_legend(title, colors_by_value):
    rows = []
    for value, color in colors_by_value.items():
        rows.append(
            '<div class="legend-row">'
            f'<span class="swatch" style="background:{color}"></span>'
            f"{html.escape(str(value))}</div>"
        )
    return (
        f'<div class="legend"><div class="legend-title">{html.escape(title)}</div>'
        f'<div class="legend-rows">{"".join(rows)}</div></div>'
    )


def _legend_card(layer_name, content, visible, map_id="map", description=""):
    style = "" if visible else ' style="display:none"'
    description_html = (
        f'<p class="legend-note">{html.escape(description)}</p>'
        if description
        else ""
    )
    return (
        f'<div class="legend-card" data-layer="{html.escape(layer_name)}" '
        f'data-map="{html.escape(map_id)}"{style}>'
        f"{content}{description_html}</div>"
    )


SOCIAL_GROUP_LAYERS = [
    ("service_access_education_min", "Доступность: образование", "Образование, мин", ["#eff3ff", "#6baed6", "#08519c"]),
    ("service_access_healthcare_min", "Доступность: здравоохранение", "Здравоохранение, мин", ["#f7fcf5", "#74c476", "#00441b"]),
    ("service_access_social_sphere_min", "Доступность: социальная сфера", "Социальная сфера, мин", ["#f2f0f7", "#9e9ac8", "#54278f"]),
    ("service_access_culture_min", "Доступность: культура", "Культура, мин", ["#fff7f3", "#f768a1", "#7a0177"]),
    ("service_access_recreation_min", "Доступность: рекреация", "Рекреация, мин", ["#f7fcb9", "#addd8e", "#006837"]),
]
CONSUMER_GROUP_LAYERS = [
    ("service_access_retail_min", "Доступность: торговля", "Торговля, мин", ["#fff5eb", "#fd8d3c", "#7f2704"]),
    ("service_access_food_min", "Доступность: общепит", "Общепит, мин", ["#fef0d9", "#fc8d59", "#b30000"]),
    ("service_access_tourism_min", "Доступность: туризм", "Туризм, мин", ["#f7fcf0", "#7bccc4", "#084081"]),
    ("service_access_household_services_min", "Доступность: услуги", "Бытовые услуги, мин", ["#f7fcfd", "#66c2a4", "#00441b"]),
    ("service_access_transport_min", "Доступность: транспорт", "Транспорт, мин", ["#f2f0f7", "#bcbddc", "#3f007d"]),
]
SERVICE_GROUP_LABELS = {
    "education": "Образование",
    "healthcare": "Здравоохранение",
    "social_sphere": "Социальная сфера",
    "culture": "Культура",
    "recreation": "Рекреация",
    "retail": "Торговля",
    "food": "Общепит",
    "tourism": "Туризм",
    "household_services": "Услуги",
    "transport": "Транспорт",
    "infrastructure": "Инфраструктура",
    "industry": "Промышленность",
    "nature": "Природа",
    "special": "Специальное",
    "other": "Прочее",
}
SERVICE_GROUP_BY_LABEL = {
    label.lower(): label.lower() for label in SERVICE_GROUP_LABELS.values()
}
SERVICE_TYPE_LABELS = {
    "school": "школа",
    "kindergarten": "детский сад",
    "college": "колледж",
    "university": "вуз",
    "extra_education": "дополнительное образование",
    "hospital": "больница",
    "polyclinic": "поликлиника",
    "women_clinic": "женская консультация",
    "dentist": "стоматология",
    "pharmacy": "аптека",
    "rehabilitation": "реабилитация",
    "emergency": "скорая помощь",
    "orphanage": "детский дом",
    "social_facility": "социальное учреждение",
    "convenience": "продуктовый магазин",
    "supermarket": "супермаркет",
    "mall": "торговый центр",
    "marketplace": "торговая площадка",
    "market": "рынок",
    "clothes": "магазин одежды",
    "electronic": "электроника",
    "household": "товары для дома",
    "houseware": "посуда и хозтовары",
    "bakery": "булочная",
    "cafe": "кафе",
    "restaurant": "ресторан",
    "bar": "бар",
    "bank": "банк",
    "lawyer": "юрист",
    "notary": "нотариус",
    "beauty": "салон красоты",
    "hairdresser": "парикмахерская",
    "post_office": "почта",
    "community_centre": "общественный центр",
    "mfz": "МФЦ",
    "library": "библиотека",
    "museum": "музей",
    "theatre": "театр",
    "cinema": "кинотеатр",
    "religion": "религиозный объект",
    "historic": "исторический объект",
    "hotel": "гостиница",
    "hostel": "хостел",
    "guest_house": "гостевой дом",
    "bus_stop": "автобусная остановка",
    "bus_station": "автовокзал",
    "train_station": "железнодорожная станция",
    "train_building": "здание вокзала",
    "parking": "парковка",
    "fuel": "АЗС",
    "pier": "причал",
    "park": "парк",
    "beach": "пляж",
    "embankment": "набережная",
    "pitch": "спортивная площадка",
    "playground": "детская площадка",
    "stadium": "стадион",
    "swimming_pool": "бассейн",
    "sport_centre": "спортивный центр",
    "forest": "лес",
    "farmland": "сельхозугодья",
    "substation": "подстанция",
    "water_works": "водозабор",
    "wastewater_plant": "очистные сооружения",
    "boilerhouse": "котельная",
    "pumping_station": "насосная станция",
    "cell_tower": "вышка связи",
    "engineering": "инженерный объект",
    "hydroelectric_pp": "ГЭС",
    "cemetery": "кладбище",
    "police": "полиция",
    "fire_station": "пожарная часть",
    "outpost": "КПП / погранзастава",
    "recruitment": "военкомат",
    "construction": "стройка",
    "chemical": "химическое производство",
    "food_industry": "пищевое производство",
    "light_industry": "лёгкая промышленность",
    "veterinary": "ветеринарная клиника",
}
LAYER_DESCRIPTIONS = {
    "Обеспеченность школами": "Доля школьного спроса квартала, покрытая местами в школах в пределах 15 минут. 1 — спрос покрыт полностью, 0 — не покрыт.",
    "Обеспеченность слабая": "Более мягкий вариант распределения школьных мест: показывает покрытие спроса без жёсткой конкуренции за места.",
    "Обеспеченность детсадами": "Доля спроса на детские сады, покрытая местами в пределах 7 минут. 1 — спрос покрыт полностью, 0 — не покрыт.",
    "Обеспеченность детсадами (слабая)": "Более мягкий вариант распределения мест в детсадах: показывает покрытие спроса без жёсткой конкуренции.",
    "Социальные сервисы": "Число объектов социальной сферы в границах квартала. В сферу входят: образование, здравоохранение, социальная сфера, культура и рекреация.",
    "Плотность: социальные сервисы": "Сколько социальных объектов приходится на 1 м² площади квартала. Позволяет сравнивать кварталы разного размера.",
    "Разнообразие: социальные сервисы": "Индекс Шеннона по типам социальных объектов: 0 — один тип, чем выше, тем больше разных типов.",
    "Доступность: социальные сервисы": "Время пешком до ближайшего социального объекта любой группы. В сферу входят: образование, здравоохранение, социальная сфера, культура и рекреация.",
    "В 15 минутах: социальные сервисы": "Сколько социальных объектов находится в 15-минутной пешей зоне от квартала.",
    "Коммерческие и бытовые сервисы": "Число объектов коммерческой и бытовой сферы в границах квартала. В сферу входят: торговля, общепит, туризм, услуги и транспорт.",
    "Плотность: коммерческие и бытовые сервисы": "Сколько коммерческих и бытовых объектов приходится на 1 м² площади квартала.",
    "Разнообразие: коммерческие и бытовые сервисы": "Индекс Шеннона по типам коммерческих объектов: 0 — один тип, чем выше, тем больше разных типов.",
    "Доступность: коммерческие и бытовые сервисы": "Время пешком до ближайшего коммерческого или бытового объекта любой группы. В сферу входят: торговля, общепит, туризм, услуги и транспорт.",
    "В 15 минутах: коммерческие и бытовые сервисы": "Сколько коммерческих и бытовых объектов находится в 15-минутной пешей зоне от квартала.",
    "Землепользование": "Основная функциональная зона квартала по площади: жилая, деловая, рекреационная, промышленная и т.д.",
    "Морфотипы Spacematrix": "Модельный тип застройки BlocksNet по плотности, этажности и смешанности. Это не официальная типология.",
    "Морфотипы Strelka": "Модельный морфотип Strelka, описывающий характер застройки квартала.",
    "Зональная доступность": "Среднее время пешком от квартала до всей территории, взвешенное по площади. Меньше — квартал ближе к остальным.",
    "Население": "Сумма населения зданий в квартале по готовому атрибуту population.",
    "Спрос на школы": "Расчётная потребность в школьных местах: население квартала умножается на норматив.",
    "Плотность FSI": "Отношение площади этажей к площади квартала. Показывает, насколько интенсивно используется территория.",
    "Плотность GSI": "Доля площади квартала, занятая застройкой. Показывает плотность пятна застройки.",
    "Этажность L": "Средняя этажность застройки квартала.",
    "Смешанность MXI": "Насколько в квартале смешаны разные функции: 0 — одна функция, выше — больше смешения.",
    "OSR": "Отношение площади озеленённых и открытых пространств к площади застройки. Выше — больше открытых пространств.",
    "Образование": "Школы, детские сады, колледжи, вузы и дополнительное образование.",
    "Здравоохранение": "Больницы, поликлиники, женские консультации, стоматологии, аптеки, реабилитация и скорая помощь.",
    "Социальная сфера": "Детские дома и социальные учреждения.",
    "Культура": "Библиотеки, музеи, театры, кинотеатры, религиозные и исторические объекты.",
    "Рекреация": "Парки, пляжи, набережные, спортплощадки, детские площадки, стадионы, бассейны и спортцентры.",
    "Торговля": "Продуктовые магазины, супермаркеты, торговые центры, рынки, булочная, одежда, электроника и хозтовары.",
    "Общепит": "Кафе, бары и рестораны; булочная в текущей классификации отнесена к торговле.",
    "Туризм": "Гостиницы, хостелы и гостевые дома.",
    "Услуги": "Банки, парикмахерские, салоны красоты, почта, МФЦ, юристы и общественные центры.",
    "Транспорт": "Автобусные остановки, парковки, АЗС, железнодорожные станции и здания вокзалов.",
    "Инфраструктура": "Котельные, подстанции, водозаборы, очистные сооружения, вышки связи и инженерные объекты.",
    "Промышленность": "Химическое, пищевое и лёгкое производство.",
    "Природа": "Леса и сельхозугодья.",
    "Специальное": "Кладбища, полиция, пожарные части, КПП и военкоматы.",
    "Прочее": "Стройки, ветеринарные клиники и объекты без отдельной группы.",
}
_GROUP_CONTENTS_CACHE = None


def service_group_contents():
    global _GROUP_CONTENTS_CACHE
    if _GROUP_CONTENTS_CACHE is not None:
        return _GROUP_CONTENTS_CACHE
    path = RESULTS_DIR / "services_by_type.csv"
    if not path.exists():
        _GROUP_CONTENTS_CACHE = {}
        return _GROUP_CONTENTS_CACHE
    frame = pd.read_csv(path)
    contents = {}
    for group, group_frame in frame.groupby("service_group", dropna=False):
        labels = [
            SERVICE_TYPE_LABELS.get(service_type, service_type.replace("_", " "))
            for service_type in group_frame["service_type"]
        ]
        contents[str(group)] = ", ".join(dict.fromkeys(labels))
    _GROUP_CONTENTS_CACHE = contents
    return contents


def layer_description(layer_name: str) -> str:
    if layer_name in LAYER_DESCRIPTIONS:
        return LAYER_DESCRIPTIONS[layer_name]
    if layer_name.startswith("Доступность: "):
        group_label = layer_name.removeprefix("Доступность: ").strip()
        group = SERVICE_GROUP_BY_LABEL.get(group_label.lower())
        if group:
            contents = service_group_contents().get(group, "")
            description = f"Время пешком до ближайшего объекта группы «{group_label.lower()}»."
            if contents:
                description += f" В группу входят: {contents}."
            if group == "общепит":
                description += " Булочная в текущей классификации отнесена к торговле."
            return description
    return ""


def build_map(frame, schools_gdf, kindergartens_gdf=None, scope="social", map_id="social"):
    def legend_card(layer_name, content, visible):
        return _legend_card(
            layer_name, content, visible, map_id, layer_description(layer_name)
        )

    gdf = frame.to_crs(4326)
    geojson_data = json.loads(gdf.to_json())
    for feature in geojson_data["features"]:
        properties = feature.get("properties", {})
        for key, value in list(properties.items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                properties[key] = None

    min_x, min_y, max_x, max_y = gdf.total_bounds
    center = [(min_y + max_y) / 2, (min_x + max_x) / 2]
    map_object = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    boundary_path = PROCESSED_DIR / "boundary.geojson"
    if boundary_path.exists():
        boundary = gpd.read_file(boundary_path).to_crs(4326)
        folium.GeoJson(
            boundary,
            name="Граница территории",
            style_function=lambda feature: {
                "fillColor": "none",
                "color": "#1f4e79",
                "weight": 2.5,
                "fillOpacity": 0.0,
            },
        ).add_to(map_object)

    for layer_name, filename, color, weight, fill_opacity in (
        ("Водные объекты", "water.geojson", "#1f78b4", 0.8, 0.35),
        ("Реки", "river.geojson", "#1f78b4", 1.5, 0.0),
    ):
        layer_path = PROCESSED_DIR / filename
        if layer_path.exists():
            layer = gpd.read_file(layer_path).to_crs(4326)
            if len(layer):
                folium.GeoJson(
                    layer,
                    name=layer_name,
                    show=False,
                    style_function=lambda feature, color=color, weight=weight, fill_opacity=fill_opacity: {
                        "fillColor": color,
                        "color": color,
                        "weight": weight,
                        "fillOpacity": fill_opacity,
                    },
                ).add_to(map_object)

    popup = [
        ("block_id", "Квартал"),
        ("land_use", "Землепользование"),
        ("population", "Население, чел."),
        ("area_accessibility", "Зональная доступность, мин"),
        ("fsi", "FSI"),
        ("gsi", "GSI"),
        ("l", "Этажность L"),
        ("mxi", "MXI"),
        ("osr", "OSR"),
        ("spacematrix_morphotype", "Морфотип Spacematrix"),
        ("strelka_morphotype", "Морфотип Strelka"),
    ]
    if scope == "social":
        popup.extend(
            [
                ("demand", "Спрос на школы, мест"),
                ("capacity", "Вместимость школ, мест"),
                ("kindergarten_capacity", "Вместимость детсадов, мест"),
                ("provision_strong", "Обеспеченность школами, 0–1"),
                ("provision_weak", "Обеспеченность школами (слабая), 0–1"),
                ("kindergarten_provision_strong", "Обеспеченность детсадами, 0–1"),
                ("kindergarten_provision_weak", "Обеспеченность детсадами (слабая), 0–1"),
                ("services_social_count", "Социальных сервисов, объектов"),
                ("services_social_diversity", "Разнообразие социальных сервисов"),
                ("service_access_social_min", "Ближайший социальный сервис, мин"),
                ("social_service_count_15min", "Социальных сервисов в 15 мин"),
            ]
        )
    else:
        popup.extend(
            [
                ("services_consumer_count", "Коммерческих и бытовых сервисов, объектов"),
                ("services_consumer_diversity", "Разнообразие коммерческих сервисов"),
                ("service_access_consumer_min", "Ближайший коммерческий сервис, мин"),
                ("consumer_service_count_15min", "Коммерческих сервисов в 15 мин"),
            ]
        )
    popup_fields = [field for field, _ in popup]
    popup_aliases = [alias for _, alias in popup]
    available_popup_fields = [
        field for field in popup_fields if field in gdf.columns or field == "block_id"
    ]
    available_popup_aliases = [
        alias
        for field, alias in zip(popup_fields, popup_aliases)
        if field in available_popup_fields
    ]
    def make_popup():
        return folium.GeoJsonPopup(
            fields=available_popup_fields,
            aliases=available_popup_aliases,
            localize=True,
        )

    def add_choropleth(
        field,
        name,
        label,
        colors,
        minimum,
        maximum,
        show=True,
        tooltip_field=None,
    ):
        colormap = LinearColormap(colors=colors, vmin=minimum, vmax=maximum)
        tooltip_field = tooltip_field or field

        def style_function(feature):
            value = feature["properties"].get(field)
            if value is None:
                return {
                    "fillColor": "#d9d9d9",
                    "color": "#666666",
                    "weight": 0.4,
                    "fillOpacity": 0.6,
                }
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return {
                    "fillColor": "#d9d9d9",
                    "color": "#666666",
                    "weight": 0.4,
                    "fillOpacity": 0.6,
                }
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return {
                    "fillColor": "#d9d9d9",
                    "color": "#666666",
                    "weight": 0.4,
                    "fillOpacity": 0.6,
                }
            return {
                "fillColor": colormap(numeric_value),
                "color": "#666666",
                "weight": 0.4,
                "fillOpacity": 0.78,
            }

        layer = folium.GeoJson(
            geojson_data,
            name=name,
            show=show,
            style_function=style_function,
            highlight_function=lambda feature: {
                "weight": 1.6,
                "color": "#111111",
                "fillOpacity": 0.9,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["block_id", tooltip_field],
                aliases=["Квартал", label],
                localize=True,
                sticky=False,
            ),
            popup=make_popup(),
        )
        layer.add_to(map_object)
        legend_layers.append((name, layer.get_name()))
        return colormap

    numeric_provision = pd.to_numeric(frame.get("provision_strong"), errors="coerce").dropna()
    numeric_provision_weak = pd.to_numeric(frame.get("provision_weak"), errors="coerce").dropna()
    numeric_kindergarten = pd.to_numeric(
        frame.get("kindergarten_provision_strong"), errors="coerce"
    ).dropna()
    numeric_kindergarten_weak = pd.to_numeric(
        frame.get("kindergarten_provision_weak"), errors="coerce"
    ).dropna()
    service_count_field = f"services_{scope}_count"
    service_density_field = f"services_{scope}_density"
    service_diversity_field = f"services_{scope}_diversity"
    service_access_field = f"service_access_{scope}_min"
    service_count_15_field = f"{scope}_service_count_15min"
    numeric_services = pd.to_numeric(
        frame.get(service_count_field), errors="coerce"
    ).dropna()
    numeric_service_density = pd.to_numeric(
        frame.get(service_density_field), errors="coerce"
    ).dropna()
    numeric_diversity = pd.to_numeric(
        frame.get(service_diversity_field), errors="coerce"
    ).dropna()
    numeric_service_access = pd.to_numeric(
        frame.get(service_access_field), errors="coerce"
    ).dropna()
    numeric_service_count_15 = pd.to_numeric(
        frame.get(service_count_15_field), errors="coerce"
    ).dropna()
    numeric_area = pd.to_numeric(frame.get("area_accessibility"), errors="coerce").dropna()
    numeric_population = pd.to_numeric(frame.get("population"), errors="coerce").dropna()
    numeric_demand = pd.to_numeric(frame.get("demand"), errors="coerce").dropna()
    numeric_fsi = pd.to_numeric(frame.get("fsi"), errors="coerce").dropna()
    numeric_gsi = pd.to_numeric(frame.get("gsi"), errors="coerce").dropna()
    numeric_l = pd.to_numeric(frame.get("l"), errors="coerce").dropna()
    numeric_mxi = pd.to_numeric(frame.get("mxi"), errors="coerce").dropna()
    numeric_osr = (
        pd.to_numeric(frame.get("osr"), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    numeric_osr = numeric_osr[numeric_osr > 0]
    osr_min = float(numeric_osr.min()) if len(numeric_osr) else None
    osr_max = float(numeric_osr.max()) if len(numeric_osr) else None
    for feature in geojson_data["features"]:
        properties = feature.get("properties", {})
        value = properties.get("osr")
        try:
            numeric_value = float(value) if value is not None else None
        except (TypeError, ValueError):
            numeric_value = None
        if (
            numeric_value is not None
            and math.isfinite(numeric_value)
            and numeric_value > 0
        ):
            properties["__osr_log"] = math.log10(numeric_value)
        else:
            properties["__osr_log"] = None

    legends = []
    legend_layers = []
    if scope == "social" and "provision_strong" in frame.columns and len(numeric_provision):
        colormap = add_choropleth(
            "provision_strong",
            "Обеспеченность школами",
            "Обеспеченность сильная, 0–1",
            ["#d73027", "#fee08b", "#1a9850"],
            0.0,
            1.0,
            show=True,
        )
        legends.append(
            legend_card(
                "Обеспеченность школами",
                gradient_legend(["#d73027", "#fee08b", "#1a9850"], 0.0, 1.0, "Обеспеченность школами, 0–1"),
                True,
            )
        )
    if scope == "social" and "provision_weak" in frame.columns and len(numeric_provision_weak):
        add_choropleth(
            "provision_weak",
            "Обеспеченность слабая",
            "Обеспеченность слабая, 0–1",
            ["#d73027", "#fee08b", "#1a9850"],
            0.0,
            1.0,
            show=False,
        )
        legends.append(
            legend_card(
                "Обеспеченность слабая",
                gradient_legend(["#d73027", "#fee08b", "#1a9850"], 0.0, 1.0, "Обеспеченность слабая, 0–1"),
                False,
            )
        )
    if scope == "social" and "kindergarten_provision_strong" in frame.columns and len(numeric_kindergarten):
        add_choropleth(
            "kindergarten_provision_strong",
            "Обеспеченность детсадами",
            "Обеспеченность детсадами, 0–1",
            ["#d73027", "#fee08b", "#1a9850"],
            0.0,
            1.0,
            show=False,
        )
        legends.append(
            legend_card(
                "Обеспеченность детсадами",
                gradient_legend(["#d73027", "#fee08b", "#1a9850"], 0.0, 1.0, "Обеспеченность детсадами, 0–1"),
                False,
            )
        )
    if scope == "social" and "kindergarten_provision_weak" in frame.columns and len(numeric_kindergarten_weak):
        add_choropleth(
            "kindergarten_provision_weak",
            "Обеспеченность детсадами (слабая)",
            "Обеспеченность детсадами слабая, 0–1",
            ["#d73027", "#fee08b", "#1a9850"],
            0.0,
            1.0,
            show=False,
        )
        legends.append(
            legend_card(
                "Обеспеченность детсадами (слабая)",
                gradient_legend(["#d73027", "#fee08b", "#1a9850"], 0.0, 1.0, "Обеспеченность детсадами слабая, 0–1"),
                False,
            )
        )
    scope_label = (
        "Социальные сервисы" if scope == "social" else "Коммерческие и бытовые сервисы"
    )
    if service_count_field in frame.columns and len(numeric_services):
        add_choropleth(
            service_count_field,
            scope_label,
            "Объектов",
            ["#f7fcf5", "#74c476", "#00441b"],
            0.0,
            float(numeric_services.max()),
            show=False,
        )
        legends.append(
            legend_card(
                scope_label,
                gradient_legend(
                    ["#f7fcf5", "#74c476", "#00441b"],
                    0.0,
                    float(numeric_services.max()),
                    "Объектов",
                ),
                False,
            )
        )
    if service_density_field in frame.columns and len(numeric_service_density):
        density_name = f"Плотность: {scope_label.lower()}"
        add_choropleth(
            service_density_field,
            density_name,
            "Объектов на м²",
            ["#f7fbff", "#6baed6", "#08306b"],
            0.0,
            float(numeric_service_density.max()),
            show=False,
        )
        legends.append(
            legend_card(
                density_name,
                gradient_legend(
                    ["#f7fbff", "#6baed6", "#08306b"],
                    0.0,
                    float(numeric_service_density.max()),
                    "Объектов на м²",
                ),
                False,
            )
        )
    if service_diversity_field in frame.columns and len(numeric_diversity):
        diversity_name = f"Разнообразие: {scope_label.lower()}"
        add_choropleth(
            service_diversity_field,
            diversity_name,
            "Индекс Шеннона",
            ["#fcfbfd", "#9e9ac8", "#3f007d"],
            0.0,
            float(numeric_diversity.max()),
            show=False,
        )
        legends.append(
            legend_card(
                diversity_name,
                gradient_legend(
                    ["#fcfbfd", "#9e9ac8", "#3f007d"],
                    0.0,
                    float(numeric_diversity.max()),
                    "Индекс Шеннона",
                ),
                False,
            )
        )
    if service_access_field in frame.columns and len(numeric_service_access):
        access_name = f"Доступность: {scope_label.lower()}"
        add_choropleth(
            service_access_field,
            access_name,
            "Ближайший объект, мин",
            ["#1a9850", "#fee08b", "#d73027"],
            0.0,
            float(numeric_service_access.max()),
            show=scope == "consumer",
        )
        legends.append(
            legend_card(
                access_name,
                gradient_legend(
                    ["#1a9850", "#fee08b", "#d73027"],
                    0.0,
                    float(numeric_service_access.max()),
                    "Ближайший объект, мин",
                ),
                scope == "consumer",
            )
        )
    if service_count_15_field in frame.columns and len(numeric_service_count_15):
        count15_name = f"В 15 минутах: {scope_label.lower()}"
        add_choropleth(
            service_count_15_field,
            count15_name,
            "Объектов в 15 мин",
            ["#f7fcfd", "#66c2a4", "#00441b"],
            0.0,
            float(numeric_service_count_15.max()),
            show=False,
        )
        legends.append(
            legend_card(
                count15_name,
                gradient_legend(
                    ["#f7fcfd", "#66c2a4", "#00441b"],
                    0.0,
                    float(numeric_service_count_15.max()),
                    "Объектов в 15 мин",
                ),
                False,
            )
        )
    group_layers = SOCIAL_GROUP_LAYERS if scope == "social" else CONSUMER_GROUP_LAYERS
    for field, name, label, colors in group_layers:
        values = pd.to_numeric(frame.get(field), errors="coerce").dropna()
        if not len(values):
            continue
        maximum = max(float(values.max()), 1.0)
        add_choropleth(field, name, label, colors, 0.0, maximum, show=False)
        legends.append(
            legend_card(
                name,
                gradient_legend(colors, 0.0, maximum, label),
                False,
            )
        )
    if "area_accessibility" in frame.columns and len(numeric_area):
        add_choropleth(
            "area_accessibility",
            "Зональная доступность",
            "Зональная доступность, мин",
            ["#1a9850", "#fee08b", "#d73027"],
            float(numeric_area.min()),
            float(numeric_area.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Зональная доступность",
                gradient_legend(["#1a9850", "#fee08b", "#d73027"], float(numeric_area.min()), float(numeric_area.max()), "Зональная доступность, мин"),
                False,
            )
        )
    if "population" in frame.columns and len(numeric_population):
        add_choropleth(
            "population",
            "Население",
            "Население, чел.",
            ["#ffffcc", "#fd8d3c", "#800026"],
            0.0,
            float(numeric_population.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Население",
                gradient_legend(["#ffffcc", "#fd8d3c", "#800026"], 0.0, float(numeric_population.max()), "Население, чел."),
                False,
            )
        )
    if scope == "social" and "demand" in frame.columns and len(numeric_demand):
        add_choropleth(
            "demand",
            "Спрос на школы",
            "Спрос на школы, мест",
            ["#fff5eb", "#fd8d3c", "#7f2704"],
            0.0,
            float(numeric_demand.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Спрос на школы",
                gradient_legend(["#fff5eb", "#fd8d3c", "#7f2704"], 0.0, float(numeric_demand.max()), "Спрос на школы, мест"),
                False,
            )
        )
    if "fsi" in frame.columns and len(numeric_fsi):
        add_choropleth(
            "fsi",
            "Плотность FSI",
            "FSI",
            ["#f7fcf5", "#74c476", "#00441b"],
            0.0,
            float(numeric_fsi.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Плотность FSI",
                gradient_legend(["#f7fcf5", "#74c476", "#00441b"], 0.0, float(numeric_fsi.max()), "FSI"),
                False,
            )
        )
    if "gsi" in frame.columns and len(numeric_gsi):
        add_choropleth(
            "gsi",
            "Плотность GSI",
            "GSI",
            ["#f7fbff", "#6baed6", "#08306b"],
            0.0,
            float(numeric_gsi.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Плотность GSI",
                gradient_legend(["#f7fbff", "#6baed6", "#08306b"], 0.0, float(numeric_gsi.max()), "GSI"),
                False,
            )
        )
    if "l" in frame.columns and len(numeric_l):
        add_choropleth(
            "l",
            "Этажность L",
            "Этажность L",
            ["#fcfbfd", "#9e9ac8", "#3f007d"],
            float(numeric_l.min()),
            float(numeric_l.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Этажность L",
                gradient_legend(["#fcfbfd", "#9e9ac8", "#3f007d"], float(numeric_l.min()), float(numeric_l.max()), "Этажность L"),
                False,
            )
        )
    if "mxi" in frame.columns and len(numeric_mxi):
        add_choropleth(
            "mxi",
            "Смешанность MXI",
            "MXI",
            ["#f7fcfd", "#66c2a4", "#00441b"],
            0.0,
            float(numeric_mxi.max()),
            show=False,
        )
        legends.append(
            legend_card(
                "Смешанность MXI",
                gradient_legend(["#f7fcfd", "#66c2a4", "#00441b"], 0.0, float(numeric_mxi.max()), "MXI"),
                False,
            )
        )
    if osr_min is not None and osr_max is not None:
        add_choropleth(
            "__osr_log",
            "OSR",
            "OSR",
            ["#fff7ec", "#d95f0e", "#7f0000"],
            math.log10(osr_min),
            math.log10(osr_max),
            show=False,
            tooltip_field="osr",
        )
        legends.append(
            legend_card(
                "OSR",
                gradient_legend(["#fff7ec", "#d95f0e", "#7f0000"], osr_min, osr_max, "OSR (логарифмическая шкала)"),
                False,
            )
        )

    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
        "#eeca3b",
    ]

    def add_categorical(field, name, title, show=False):
        if field not in frame.columns:
            return
        categories = [
            value
            for value in frame[field].dropna().unique().tolist()
            if str(value).strip()
        ]
        categories = sorted(categories, key=lambda value: str(value))
        colors_by_value = {
            value: palette[index % len(palette)] for index, value in enumerate(categories)
        }
        colors_by_value["нет данных"] = "#d9d9d9"

        def style_function(feature):
            value = feature["properties"].get(field)
            if value is None or str(value).strip() == "":
                value = "нет данных"
            color = colors_by_value.get(value, "#d9d9d9")
            return {
                "fillColor": color,
                "color": "#666666",
                "weight": 0.4,
                "fillOpacity": 0.78,
            }

        layer = folium.GeoJson(
            geojson_data,
            name=name,
            show=show,
            style_function=style_function,
            highlight_function=lambda feature: {
                "weight": 1.6,
                "color": "#111111",
                "fillOpacity": 0.9,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["block_id", field],
                aliases=["Квартал", title],
                localize=True,
                sticky=False,
            ),
            popup=make_popup(),
        )
        layer.add_to(map_object)
        legend_layers.append((name, layer.get_name()))
        legends.append(legend_card(name, categorical_legend(title, colors_by_value), show))

    add_categorical("land_use", "Землепользование", "Землепользование")
    add_categorical("spacematrix_morphotype", "Морфотипы Spacematrix", "Морфотипы Spacematrix")
    add_categorical("strelka_morphotype", "Морфотипы Strelka", "Морфотипы Strelka")

    for layer_name, filename, color, weight, dash_array in (
        ("Дороги", "roads.geojson", "#555555", 0.9, None),
        ("Железные дороги", "railways.geojson", "#333333", 1.3, "5,4"),
    ):
        layer_path = PROCESSED_DIR / filename
        if layer_path.exists():
            layer = gpd.read_file(layer_path).to_crs(4326)
            if len(layer):
                folium.GeoJson(
                    layer,
                    name=layer_name,
                    show=False,
                    style_function=lambda feature, color=color, weight=weight, dash_array=dash_array: {
                        "color": color,
                        "weight": weight,
                        "opacity": 0.65,
                        "dashArray": dash_array,
                    },
                ).add_to(map_object)

    if scope == "social":
        schools_group = folium.FeatureGroup(name="Школы", show=True)
        if schools_gdf is not None and len(schools_gdf):
            schools = schools_gdf.to_crs(4326).copy()
            for _, row in schools.iterrows():
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    continue
                point = geometry.centroid
                name = row.get("name")
                if not isinstance(name, str) or not name.strip():
                    name = "Школа"
                capacity = pd.to_numeric(row.get("capacity"), errors="coerce")
                capacity_text = (
                    f"Вместимость: {capacity:,.0f} мест (исходные данные)."
                    if pd.notna(capacity)
                    else "Вместимость в исходных данных не указана."
                )
                folium.Marker(
                    location=[point.y, point.x],
                    tooltip=html.escape(name),
                    popup=folium.Popup(
                        f"<b>{html.escape(name)}</b><br>{html.escape(capacity_text)}",
                        max_width=320,
                    ),
                    icon=folium.Icon(color="green", icon="graduation-cap", prefix="fa"),
                ).add_to(schools_group)
        schools_group.add_to(map_object)

        kindergartens_group = folium.FeatureGroup(name="Детские сады", show=True)
        if kindergartens_gdf is not None and len(kindergartens_gdf):
            kindergartens = kindergartens_gdf.to_crs(4326).copy()
            for _, row in kindergartens.iterrows():
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    continue
                point = geometry.centroid
                name = row.get("name")
                if not isinstance(name, str) or not name.strip():
                    name = "Детский сад"
                capacity = pd.to_numeric(row.get("capacity"), errors="coerce")
                capacity_text = (
                    f"Вместимость: {capacity:,.0f} мест (исходные данные)."
                    if pd.notna(capacity)
                    else "Вместимость в исходных данных не указана."
                )
                folium.Marker(
                    location=[point.y, point.x],
                    tooltip=html.escape(name),
                    popup=folium.Popup(
                        f"<b>{html.escape(name)}</b><br>{html.escape(capacity_text)}",
                        max_width=320,
                    ),
                    icon=folium.Icon(color="orange", icon="child", prefix="fa"),
                ).add_to(kindergartens_group)
        kindergartens_group.add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)

    map_object.fit_bounds([[min_y, min_x], [max_y, max_x]])

    legend_map_js = ",\n".join(
        f"      {json.dumps(name, ensure_ascii=False)}: {variable}"
        for name, variable in legend_layers
    )
    map_variable = map_object.get_name()
    legend_script = (
        "<script>\n"
        "window.addEventListener('load', function () {\n"
        f"  var mapObject = window[{json.dumps(map_variable)}];\n"
        "  if (!mapObject) { return; }\n"
        "  var legendLayers = {\n"
        f"{legend_map_js}\n"
        "  };\n"
        "  function postLegendState() {\n"
        "    var active = [];\n"
        "    for (var layerName in legendLayers) {\n"
        "      if (mapObject.hasLayer(legendLayers[layerName])) {\n"
        "        active.push(layerName);\n"
        "      }\n"
        "    }\n"
        "    window.parent.postMessage({type: 'legend-state', map: " + json.dumps(map_id) + ", active: active}, '*');\n"
        "  }\n"
        "  mapObject.on('overlayadd', postLegendState);\n"
        "  mapObject.on('overlayremove', postLegendState);\n"
        "  postLegendState();\n"
        "});\n"
        "</script>"
    )
    map_full = map_object.get_root().render()
    if "</html>" in map_full:
        map_full = map_full.replace("</html>", legend_script + "\n</html>", 1)
    else:
        map_full += legend_script
    map_html = (
        f'<iframe srcdoc="{html.escape(map_full)}" width="100%" height="100%" '
        'style="border:0;" allowfullscreen></iframe>'
    )
    return map_html, "\n".join(legends)


def main():
    frame = load_metrics()
    schools_gdf = None
    schools_path = PROCESSED_DIR / "schools.geojson"
    if schools_path.exists():
        schools_gdf = gpd.read_file(schools_path)

    kindergartens_gdf = None
    kindergartens_path = PROCESSED_DIR / "kindergartens.geojson"
    if kindergartens_path.exists():
        kindergartens_gdf = gpd.read_file(kindergartens_path)

    social_map_html, social_legends_html = build_map(
        frame, schools_gdf, kindergartens_gdf, scope="social", map_id="social"
    )
    consumer_map_html, consumer_legends_html = build_map(
        frame, None, None, scope="consumer", map_id="consumer"
    )
    charts_html = make_charts(frame)

    population = pd.to_numeric(frame.get("population"), errors="coerce").fillna(0).sum()
    capacity = pd.to_numeric(frame.get("capacity"), errors="coerce").fillna(0).sum()
    kindergarten_capacity = pd.to_numeric(
        frame.get("kindergarten_capacity"), errors="coerce"
    ).fillna(0).sum()
    schools_count = pd.to_numeric(frame.get("schools_count"), errors="coerce").fillna(0).sum()
    kindergartens_count = pd.to_numeric(
        frame.get("kindergartens_count"), errors="coerce"
    ).fillna(0).sum()
    provision_values = pd.to_numeric(frame.get("provision_strong"), errors="coerce").dropna()
    mean_provision = provision_values.mean() if len(provision_values) else float("nan")
    demand_total = pd.to_numeric(frame.get("demand"), errors="coerce").fillna(0).sum()
    demand_within = pd.to_numeric(frame.get("demand_within"), errors="coerce").fillna(0).sum()
    weighted_provision = demand_within / demand_total if demand_total else float("nan")
    demand_blocks = int((pd.to_numeric(frame.get("demand"), errors="coerce").fillna(0) > 0).sum())
    zero_provision_blocks = int(
        (
            (pd.to_numeric(frame.get("demand"), errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(frame.get("provision_strong"), errors="coerce").fillna(0) == 0)
        ).sum()
    )

    kindergarten_provision_values = pd.to_numeric(
        frame.get("kindergarten_provision_strong"), errors="coerce"
    ).dropna()
    kindergarten_mean_provision = (
        kindergarten_provision_values.mean() if len(kindergarten_provision_values) else float("nan")
    )
    kindergarten_demand = pd.to_numeric(
        frame.get("kindergarten_demand"), errors="coerce"
    ).fillna(0).sum()
    kindergarten_demand_within = pd.to_numeric(
        frame.get("kindergarten_demand_within"), errors="coerce"
    ).fillna(0).sum()
    kindergarten_weighted_provision = (
        kindergarten_demand_within / kindergarten_demand if kindergarten_demand else float("nan")
    )
    services_total = pd.to_numeric(frame.get("services_count"), errors="coerce").fillna(0).sum()
    services_social_total = pd.to_numeric(
        frame.get("services_social_count"), errors="coerce"
    ).fillna(0).sum()
    services_consumer_total = pd.to_numeric(
        frame.get("services_consumer_count"), errors="coerce"
    ).fillna(0).sum()
    social_access_median = pd.to_numeric(
        frame.get("service_access_social_min"), errors="coerce"
    ).median()
    consumer_access_median = pd.to_numeric(
        frame.get("service_access_consumer_min"), errors="coerce"
    ).median()
    service_group_summary = pd.DataFrame()
    service_group_summary_path = RESULTS_DIR / "service_group_summary.csv"
    if service_group_summary_path.exists():
        service_group_summary = pd.read_csv(service_group_summary_path)

    accessibility_summary = pd.DataFrame()
    accessibility_path = RESULTS_DIR / "accessibility_summary_walk.csv"
    if accessibility_path.exists():
        accessibility_summary = pd.read_csv(accessibility_path)
    median_time = float("nan")
    if len(accessibility_summary) and "Медиана, мин" in accessibility_summary.columns:
        median_time = float(accessibility_summary["Медиана, мин"].iloc[0])

    area_median = pd.to_numeric(frame.get("area_accessibility"), errors="coerce").median()

    kpi_items = [
        ("Кварталов", fmt(len(frame)), "Готовые блоки из blocks.geojson", None),
        ("Население, чел.", fmt(int(round(population))), "Сумма атрибута population зданий", None),
        ("Вместимость школ, мест", fmt(int(round(capacity))), f"Школ в данных: {schools_count:.0f}", None),
        ("Вместимость детсадов, мест", fmt(int(round(kindergarten_capacity))), f"Детских садов: {kindergartens_count:.0f}", None),
        ("Медианное время в пути", f"{median_time:.1f} мин" if math.isfinite(median_time) else "—", "Пешеходный граф roads.geojson, 5 км/ч", None),
        ("Обеспеченность школами", f"{mean_provision:.3f}" if math.isfinite(mean_provision) else "—", f"Среднее по {len(provision_values)} кварталам", None),
        ("Покрытие школьного спроса", f"{weighted_provision:.1%}" if math.isfinite(weighted_provision) else "—", "demand_within / demand", None),
        ("Обеспеченность детсадами", f"{kindergarten_mean_provision:.3f}" if math.isfinite(kindergarten_mean_provision) else "—", f"Покрытие спроса {kindergarten_weighted_provision:.1%}" if math.isfinite(kindergarten_weighted_provision) else "Нет данных", None),
        ("Кварталов с нулевой обеспеченностью", f"{zero_provision_blocks} из {demand_blocks}", "Есть школьный спрос, но нет покрытия в 15 мин", None),
        ("Сервисов, объектов", fmt(services_total), "Все объекты в границах кварталов", None),
        ("Социальные / коммерческие", f"{fmt(services_social_total)} / {fmt(services_consumer_total)}", "Остальные — инфраструктура, природа и др.", None),
        ("Медиана доступа: соц. / комм.", f"{fmt(social_access_median, 1)} / {fmt(consumer_access_median, 1)} мин", "До ближайшего объекта сферы", None),
    ]
    kpi_html = []
    for title, value, note, badge in kpi_items:
        badge_html = f'<span class="badge badge-estimate">{badge}</span>' if badge else ""
        kpi_html.append(
            '<div class="kpi-card">'
            f'<div class="kpi-title">{html.escape(title)} {badge_html}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-note">{html.escape(note)}</div>'
            "</div>"
        )
    kpi_html = "\n".join(kpi_html)

    full_labels = {
        "block_id": "Квартал",
        "source_fid": "Исходный FID",
        "land_use": "Землепользование",
        "share": "Доля функциональной зоны",
        "population": "Население, чел.",
        "footprint_area": "Площадь застройки, м²",
        "build_floor_area": "Площадь этажей, м²",
        "living_area": "Жилая площадь, м²",
        "non_living_area": "Нежилая площадь, м²",
        "is_living": "Жилых зданий, шт.",
        "number_of_floors": "Этажей суммарно",
        "buildings_count": "Зданий, шт.",
        "schools_count": "Школ, шт.",
        "capacity": "Вместимость школ, мест",
        "capacity_is_estimate": "Вместимость школ — оценка",
        "kindergartens_count": "Детсадов, шт.",
        "kindergarten_capacity": "Вместимость детсадов, мест",
        "kindergarten_capacity_is_estimate": "Вместимость детсадов — оценка",
        "services_count": "Сервисов, объектов",
        "services_density": "Плотность сервисов, объектов/м²",
        "shannon_diversity": "Разнообразие сервисов (Шеннон)",
        "site_area": "Площадь квартала, м²",
        "fsi": "FSI",
        "gsi": "GSI",
        "mxi": "MXI",
        "l": "L (этажность)",
        "osr": "OSR",
        "share_living": "Доля жилой площади",
        "share_non_living": "Доля нежилой площади",
        "spacematrix_cluster": "Кластер Spacematrix",
        "spacematrix_morphotype": "Морфотип Spacematrix",
        "strelka_morphotype": "Морфотип Strelka",
        "demand": "Спрос, мест",
        "demand_left": "Спрос не покрыт, мест",
        "demand_within": "Спрос покрыт в 15 мин, мест",
        "demand_without": "Спрос покрыт вне 15 мин, мест",
        "capacity_left": "Ёмкость не использована, мест",
        "capacity_within": "Ёмкость в 15 мин, мест",
        "capacity_without": "Ёмкость вне 15 мин, мест",
        "provision_strong": "Обеспеченность школами, 0–1",
        "provision_weak": "Обеспеченность школами (слабая), 0–1",
        "kindergarten_demand": "Спрос на детсады, мест",
        "kindergarten_demand_left": "Спрос на детсады не покрыт, мест",
        "kindergarten_demand_within": "Спрос на детсады покрыт в 7 мин, мест",
        "kindergarten_demand_without": "Спрос на детсады вне 7 мин, мест",
        "kindergarten_capacity_left": "Ёмкость детсадов не использована, мест",
        "kindergarten_capacity_within": "Ёмкость детсадов в 7 мин, мест",
        "kindergarten_capacity_without": "Ёмкость детсадов вне 7 мин, мест",
        "kindergarten_provision_strong": "Обеспеченность детсадами, 0–1",
        "kindergarten_provision_weak": "Обеспеченность детсадами (слабая), 0–1",
        "area_accessibility": "Зональная доступность, мин",
        "services_social_count": "Социальных сервисов, объектов",
        "services_social_density": "Плотность социальных сервисов",
        "services_social_diversity": "Разнообразие социальных сервисов",
        "services_consumer_count": "Коммерческих сервисов, объектов",
        "services_consumer_density": "Плотность коммерческих сервисов",
        "services_consumer_diversity": "Разнообразие коммерческих сервисов",
        "service_access_social_min": "Ближайший социальный сервис, мин",
        "social_service_count_15min": "Социальных сервисов в 15 мин",
        "service_access_consumer_min": "Ближайший коммерческий сервис, мин",
        "consumer_service_count_15min": "Коммерческих сервисов в 15 мин",
        "service_access_any_min": "Ближайший сервис любой группы, мин",
        "service_count_15min": "Всех сервисов в 15 мин",
    }
    for slug, label in SERVICE_GROUP_LABELS.items():
        access_column = f"service_access_{slug}_min"
        count_column = f"{slug}_service_count_15min"
        if access_column in frame.columns:
            full_labels[access_column] = f"Доступность: {label.lower()}, мин"
        if count_column in frame.columns:
            full_labels[count_column] = f"{label}: объектов в 15 мин"
    full_columns = [column for column in full_labels if column in frame.columns]
    full_table = render_table(frame, columns=full_columns, labels=full_labels, table_id="metricsTable")

    land_summary = frame.copy()
    land_summary["land_use"] = land_summary["land_use"].fillna("не назначено")
    land_summary = (
        land_summary.groupby("land_use", dropna=False)
        .agg(
            blocks=("block_id", "count"),
            population=("population", "sum"),
            capacity=("capacity", "sum"),
            provision=("provision_strong", "mean"),
            area_accessibility=("area_accessibility", "mean"),
        )
        .reset_index()
    )
    land_table = render_table(
        land_summary,
        columns=["land_use", "blocks", "population", "capacity", "provision", "area_accessibility"],
        labels={
            "land_use": "Землепользование",
            "blocks": "Кварталов",
            "population": "Население, чел.",
            "capacity": "Вместимость школ, мест",
            "provision": "Средняя обеспеченность",
            "area_accessibility": "Средняя доступность, мин",
        },
    )

    provision_columns = [
        column
        for column in [
            "block_id",
            "population",
            "capacity",
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
        if column in frame.columns
    ]
    provision_table = render_table(frame, columns=provision_columns, labels=full_labels)
    links = pd.read_csv(RESULTS_DIR / "provision_school_links.csv") if (RESULTS_DIR / "provision_school_links.csv").exists() else pd.DataFrame()
    links_table = render_table(
        links,
        labels={"source": "Откуда (квартал)", "target": "Куда (квартал)", "value": "Покрыто мест"},
    )

    kindergarten_provision_columns = [
        column
        for column in [
            "block_id",
            "population",
            "kindergarten_capacity",
            "kindergarten_demand",
            "kindergarten_demand_left",
            "kindergarten_demand_within",
            "kindergarten_demand_without",
            "kindergarten_capacity_left",
            "kindergarten_capacity_within",
            "kindergarten_capacity_without",
            "kindergarten_provision_strong",
            "kindergarten_provision_weak",
        ]
        if column in frame.columns
    ]
    kindergarten_provision_table = render_table(
        frame, columns=kindergarten_provision_columns, labels=full_labels
    )
    kindergarten_links = (
        pd.read_csv(RESULTS_DIR / "provision_kindergarten_links.csv")
        if (RESULTS_DIR / "provision_kindergarten_links.csv").exists()
        else pd.DataFrame()
    )
    kindergarten_links_table = render_table(
        kindergarten_links,
        labels={"source": "Откуда (квартал)", "target": "Куда (квартал)", "value": "Покрыто мест"},
    )

    clusters = read_csv_index(RESULTS_DIR / "spacematrix_clusters.csv")
    clusters = clusters.reset_index()
    clusters_table = render_table(
        clusters,
        labels={
            "cluster_id": "Кластер",
            "l": "L",
            "fsi": "FSI",
            "mxi": "MXI",
            "l_interpretation": "Этажность",
            "fsi_interpretation": "Плотность",
            "mxi_interpretation": "Назначение",
            "morphotype": "Морфотип",
        },
    )

    accessibility_table = pd.DataFrame()
    if len(accessibility_summary):
        accessibility_table = accessibility_summary.T.reset_index()
        accessibility_table.columns = ["Показатель", "Значение"]
    accessibility_html = render_table(accessibility_table)

    schools_table_html = "<p>Нет данных.</p>"
    if schools_gdf is not None and len(schools_gdf):
        schools = schools_gdf.copy()
        if "name" not in schools.columns:
            schools["name"] = "Школа"
        schools["name"] = schools["name"].fillna("Школа")
        block_capacity = with_block_index(gpd.read_file(PROCESSED_DIR / "schools_blocks.geojson"))
        block_capacity = block_capacity.reset_index()
        block_capacity = block_capacity[["block_id", "capacity", "geometry"]].rename(
            columns={"capacity": "block_capacity"}
        )
        joined = gpd.sjoin(
            schools.to_crs(4326),
            block_capacity.to_crs(4326),
            how="left",
            predicate="within",
        )
        schools_table = joined[["name", "block_id", "capacity", "block_capacity"]].copy()
        for column in ("capacity", "block_capacity"):
            schools_table[column] = pd.to_numeric(
                schools_table[column], errors="coerce"
            ).fillna(0)
        schools_table["source"] = "Исходный слой school.geojson"
        schools_table_html = render_table(
            schools_table,
            columns=["name", "block_id", "capacity", "block_capacity", "source"],
            labels={
                "name": "Школа",
                "block_id": "Квартал",
                "capacity": "Вместимость объекта, мест",
                "block_capacity": "Вместимость квартала, мест",
                "source": "Источник",
            },
        )

    kindergartens_table_html = "<p>Нет данных.</p>"
    if kindergartens_gdf is not None and len(kindergartens_gdf):
        kindergartens = kindergartens_gdf.copy()
        if "name" not in kindergartens.columns:
            kindergartens["name"] = "Детский сад"
        kindergartens["name"] = kindergartens["name"].fillna("Детский сад")
        block_capacity = with_block_index(gpd.read_file(PROCESSED_DIR / "kindergartens_blocks.geojson"))
        block_capacity = block_capacity.reset_index()
        joined = gpd.sjoin(
            kindergartens.to_crs(4326),
            block_capacity[["block_id", "kindergarten_capacity", "geometry"]].to_crs(4326),
            how="left",
            predicate="within",
        )
        kindergartens_table = joined[
            ["name", "block_id", "capacity", "kindergarten_capacity"]
        ].copy()
        for column in ("capacity", "kindergarten_capacity"):
            kindergartens_table[column] = pd.to_numeric(
                kindergartens_table[column], errors="coerce"
            ).fillna(0)
        kindergartens_table["source"] = "Исходный слой kindergarten.geojson"
        kindergartens_table_html = render_table(
            kindergartens_table,
            columns=["name", "block_id", "capacity", "kindergarten_capacity", "source"],
            labels={
                "name": "Детский сад",
                "block_id": "Квартал",
                "capacity": "Вместимость объекта, мест",
                "kindergarten_capacity": "Вместимость квартала, мест",
                "source": "Источник",
            },
        )

    services_by_type_path = RESULTS_DIR / "services_by_type.csv"
    services_table = pd.read_csv(services_by_type_path) if services_by_type_path.exists() else pd.DataFrame()
    sphere_labels = {"social": "социальная", "consumer": "коммерческая", "other": "прочая"}
    if "service_sphere" in services_table.columns:
        services_table["service_sphere"] = services_table["service_sphere"].map(
            lambda value: sphere_labels.get(value, value)
        )
    services_table_html = render_table(
        services_table,
        columns=["service_type", "service_group", "service_sphere", "objects", "capacity_sum"],
        labels={
            "service_type": "Тип сервиса",
            "service_group": "Группа",
            "service_sphere": "Сфера",
            "objects": "Объектов",
            "capacity_sum": "Суммарная ёмкость",
        },
    )

    service_groups_table = pd.DataFrame()
    services_by_group_path = RESULTS_DIR / "services_by_group.csv"
    if services_by_group_path.exists():
        service_groups_table = pd.read_csv(services_by_group_path)
        if "service_sphere" in service_groups_table.columns:
            service_groups_table["service_sphere"] = service_groups_table[
                "service_sphere"
            ].map(lambda value: sphere_labels.get(value, value))
    service_groups_table_html = render_table(
        service_groups_table,
        columns=["service_group", "service_sphere", "objects", "capacity_sum"],
        labels={
            "service_group": "Группа",
            "service_sphere": "Сфера",
            "objects": "Объектов",
            "capacity_sum": "Суммарная ёмкость",
        },
    )

    service_group_access_table = service_group_summary.copy()
    if len(service_group_access_table):
        if "service_sphere" in service_group_access_table.columns:
            service_group_access_table["service_sphere"] = service_group_access_table[
                "service_sphere"
            ].map(lambda value: sphere_labels.get(value, value))
        service_group_access_table["share_blocks_within_15"] = (
            pd.to_numeric(
                service_group_access_table["share_blocks_within_15"], errors="coerce"
            )
            * 100
        )
    service_group_access_html = render_table(
        service_group_access_table,
        columns=[
            "service_group",
            "service_sphere",
            "objects_total",
            "median_nearest_min",
            "share_blocks_within_15",
            "median_count_15",
        ],
        labels={
            "service_group": "Группа",
            "service_sphere": "Сфера",
            "objects_total": "Объектов",
            "median_nearest_min": "Медиана до ближайшего, мин",
            "share_blocks_within_15": "Доля кварталов с объектом в 15 мин, %",
            "median_count_15": "Медиана объектов в 15 мин",
        },
    )

    completeness = pd.DataFrame(
        [
            {"Показатель": "Матрица доступности, минуты", "В ноутбуке": "раздел 10", "В дашборде": "KPI и график распределения"},
            {"Показатель": "Население, площадь застройки, этажность, жилые здания", "В ноутбуке": "раздел 7", "В дашборде": "KPI, карта населения, полная таблица"},
            {"Показатель": "Школы: capacity, count", "В ноутбуке": "раздел 8", "В дашборде": "KPI, слой школ на карте, таблица школ, полная таблица"},
            {"Показатель": "Детские сады: capacity, count", "В ноутбуке": "раздел 8", "В дашборде": "KPI, слой детсадов, таблица детсадов, обеспеченность"},
            {"Показатель": "Сервисы: count, density, diversity по социальной и коммерческой сферам", "В ноутбуке": "раздел 8", "В дашборде": "KPI, две карты, графики, таблицы по типам и группам"},
            {"Показатель": "Доступность сервисов по группам: nearest_min, count_15min", "В ноутбуке": "раздел 8", "В дашборде": "график, таблица, слои групп на двух картах"},
            {"Показатель": "Землепользование: land_use, share", "В ноутбуке": "раздел 5", "В дашборде": "карта, график населения, сводка, полная таблица"},
            {"Показатель": "Плотность: site_area, fsi, gsi, mxi, l, osr, share_living, share_non_living", "В ноутбуке": "раздел 11", "В дашборде": "карты FSI, GSI, MXI, L, OSR, график FSI/GSI, полная таблица"},
            {"Показатель": "Spacematrix: cluster, morphotype", "В ноутбуке": "раздел 11", "В дашборде": "карта, график, таблица кластеров, полная таблица"},
            {"Показатель": "Strelka: morphotype", "В ноутбуке": "раздел 11", "В дашборде": "карта, полная таблица"},
            {"Показатель": "Обеспеченность: demand, capacity, demand_left, demand_within, demand_without, capacity_left, capacity_within, capacity_without, provision_strong, provision_weak", "В ноутбуке": "раздел 11", "В дашборде": "KPI, карты сильной/слабой обеспеченности и спроса, таблица, график"},
            {"Показатель": "Связи обеспечения: source, target, value", "В ноутбуке": "раздел 11", "В дашборде": "таблица связей"},
            {"Показатель": "Зональная доступность: area_accessibility", "В ноутбуке": "раздел 11", "В дашборде": "KPI, карта, график, полная таблица"},
            {"Показатель": "Контекст: дороги, железные дороги, реки, вода", "В ноутбуке": "раздел 3", "В дашборде": "переключаемые слои карты"},
            {"Показатель": "Итоговый слой blocks_metrics", "В ноутбуке": "раздел 12", "В дашборде": "полная таблица и карта"},
        ]
    )
    completeness_table = render_table(completeness)

    share_gt_one = int((pd.to_numeric(frame.get("share"), errors="coerce") > 1).sum())
    land_use_missing = int(frame["land_use"].isna().sum()) if "land_use" in frame.columns else 0
    morph_missing = (
        int(frame["spacematrix_morphotype"].isna().sum())
        if "spacematrix_morphotype" in frame.columns
        else 0
    )
    buildings_input_count = len(gpd.read_file(PROCESSED_DIR / "buildings.geojson"))
    buildings_in_blocks = int(
        pd.to_numeric(frame.get("buildings_count"), errors="coerce").fillna(0).sum()
    )
    buildings_outside = max(buildings_input_count - buildings_in_blocks, 0)
    services_input_count = len(gpd.read_file(PROCESSED_DIR / "services_all.geojson"))
    services_outside = max(services_input_count - int(round(services_total)), 0)

    conclusions = [
        f"В границе {len(frame)} кварталов, суммарное население — {fmt(int(round(population)))} человек по атрибуту population зданий.",
        f"Медианное время пешей доступности между кварталами — {fmt(median_time, 1)} мин; максимальное значение — {fmt(accessibility_summary['Максимум, мин'].iloc[0], 1) if len(accessibility_summary) and 'Максимум, мин' in accessibility_summary.columns else '—'} мин. Расчёт выполнен по графу roads.geojson.",
        f"Школ в исходных данных: {schools_count:.0f}; суммарная вместимость — {fmt(int(round(capacity)))} мест. Это не модельная оценка: поле capacity заполнено.",
        f"Детских садов: {kindergartens_count:.0f}; суммарная вместимость — {fmt(int(round(kindergarten_capacity)))} мест; средняя сильная обеспеченность — {fmt(kindergarten_mean_provision, 3)}, покрытие спроса — {fmt(kindergarten_weighted_provision, 3)}.",
        f"Средняя сильная обеспеченность школами по кварталам — {fmt(mean_provision, 3)}; по населению покрыто {fmt(weighted_provision, 3)} спроса. В {zero_provision_blocks} из {demand_blocks} кварталов со спросом покрытия в 15 минут нет.",
        f"В папке «сервисы» учтено {fmt(int(services_total))} объектов: {fmt(services_social_total)} социальных и {fmt(services_consumer_total)} коммерческих и бытовых; остальные {fmt(max(int(round(services_total - services_social_total - services_consumer_total)), 0))} относятся к инфраструктуре, природе, промышленности, специальным и прочим группам. Для каждой группы рассчитаны время до ближайшего объекта и число объектов в 15 минутах.",
        f"Медиана времени до ближайшего социального объекта — {fmt(social_access_median, 1)} мин, до коммерческого — {fmt(consumer_access_median, 1)} мин; социальная и коммерческая сферы показаны на отдельных картах.",
        f"Землепользование назначено по площадному наложению landuse.geojson; без категории осталось {land_use_missing} кварталов, без морфотипа — {morph_missing}.",
        f"Медианная зональная доступность — {fmt(area_median, 1)} мин. Значения выше 100 минут относятся к периферийным кварталам.",
    ]
    conclusions_html = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in conclusions) + "</ul>"

    sources_html = """
<ul>
  <li>Готовые векторные слои «Ивангородское городское поселение»: blocks, building, landuse, roads, railways, river, water и папка «сервисы».</li>
  <li>BlocksNet 1.0.0a9 и пример <code>examples/pipeline.ipynb</code>: https://github.com/aimclub/blocksnet</li>
  <li>IduEdu 0.4.1 для матрицы доступности: https://github.com/DDonnyy/IduEdu</li>
  <li>Подложка карты — OpenStreetMap, © OpenStreetMap contributors, ODbL 1.0: https://www.openstreetmap.org/copyright</li>
  <li>Дашборд собран из сохранённых файлов <code>results/</code> и <code>data/processed/</code>.</li>
</ul>
"""

    units = pd.DataFrame(
        [
            {"Показатель": "Население", "Единицы": "человек", "Как получено": "сумма атрибута population зданий"},
            {"Показатель": "Площади", "Единицы": "м²", "Как получено": "геометрия готовых слоёв и агрегация зданий"},
            {"Показатель": "Вместимость школ и детсадов", "Единицы": "мест", "Как получено": "готовое поле capacity в слоях school и kindergarten"},
            {"Показатель": "Время в пути", "Единицы": "минуты", "Как получено": "пешеходный граф roads.geojson, 5 км/ч"},
            {"Показатель": "Доступность сервисов по группам", "Единицы": "минуты; объектов в 15 мин", "Как получено": "кратчайший путь до ближайшего объекта группы и подсчёт объектов в 15-минутной зоне"},
            {"Показатель": "Обеспеченность", "Единицы": "доля 0–1", "Как получено": "competitive_provision"},
            {"Показатель": "FSI, GSI, MXI, L, OSR", "Единицы": "коэффициенты", "Как получено": "calculate_density_indicators"},
            {"Показатель": "Доля функциональной зоны", "Единицы": "0–1", "Как получено": "assign_land_use; у части кварталов зоны накладываются"},
        ]
    )
    units_table = render_table(units)

    limitations = [
        "Использованы готовые слои территории; отдельная загрузка OSM в основной расчёт не требуется.",
        "Кварталы-мультиполигоны приведены к Polygon (оставлена крупнейшая часть), чтобы соответствовать схеме BlocksNet.",
        f"{buildings_outside} зданий не попали в кварталы и не участвуют в агрегации; их площадь не входит в итоговые суммы.",
        f"Сервисных объектов вне кварталов: {services_outside}; они не входят в метрики обеспеченности и разнообразия.",
        "Население взято из атрибута population зданий и может отличаться от актуальной статистики.",
        "Вместимость школ и детсадов взята из исходных слоёв; для объектов без capacity применяется импутация BlocksNet.",
        f"Без морфотипа: {morph_missing} кварталов (нет застройки или нулевые площади).",
        f"Без землепользования: {land_use_missing} кварталов.",
        f"У {share_gt_one} кварталов доля функциональной зоны больше 1 — это признак наложения зон, значение требует проверки.",
        "Считается только пешеходная доступность, 5 км/ч. Автомобиль и общественный транспорт не моделируются.",
        "Доступность сервисов по группам — время до ближайшего объекта группы по пешеходному графу; порог 15 минут выбран для сопоставимости с обеспеченностью.",
        "На карты вынесены социальная и коммерческая сферы; инфраструктура, промышленность, природа, специальные и прочие группы остаются в таблицах.",
        "Если несколько объектов группы привязаны к одному узлу графа, они учитываются в count_15min по отдельности, но время у них одинаковое.",
        "Время в пути считается между ближайшими узлами дорожного графа; у кварталов, привязанных к одному узлу, оно может быть нулевым.",
        "Средняя обеспеченность по кварталам и покрытие спроса по населению — разные показатели: первый усредняет кварталы, второй взвешивает людей.",
        "OSR равен бесконечности для кварталов без зданий; в таблицах он показан как «—».",
        "Исследование учебное: результаты зависят от полноты исходных слоёв и допущений BlocksNet.",
    ]
    limitations_html = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in limitations) + "</ul>"

    validation_table = "<p>Нет данных.</p>"
    validation_path = RESULTS_DIR / "validation_summary.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation_table = render_table(validation)

    template = Template(
        """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root {
  --bg: #f5f7fa;
  --card: #ffffff;
  --text: #1f2933;
  --muted: #5c6b7a;
  --line: #d9e2ec;
  --accent: #1f4e79;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Roboto, Arial, sans-serif;
  line-height: 1.45;
}
header {
  background: linear-gradient(135deg, #1f4e79, #2f6f9f);
  color: white;
  padding: 28px 24px;
}
header h1 { margin: 0 0 8px; font-size: 28px; }
header p { margin: 4px 0; color: #e6f0f7; }
main { max-width: 1280px; margin: 0 auto; padding: 20px; }
section {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 18px;
}
h2 { margin-top: 0; font-size: 21px; }
h3 { margin-bottom: 6px; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
}
.kpi-card {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.kpi-title { color: var(--muted); font-size: 13px; min-height: 34px; }
.kpi-value { font-size: 24px; font-weight: 700; margin: 4px 0; }
.kpi-note { color: var(--muted); font-size: 12px; }
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 600;
}
.badge-estimate { background: #fff3cd; color: #7a5b00; border: 1px solid #f0d98a; }
.map-layout {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);
  gap: 16px;
  align-items: stretch;
}
.map { min-width: 0; }
.map iframe { width: 100%; height: 680px; border: 0; border-radius: 8px; }
.legend-column {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfdff;
  max-height: 680px;
  overflow-y: auto;
}
.legend-column h3 { margin-top: 0; }
@media (max-width: 900px) {
  .map-layout { grid-template-columns: 1fr; }
  .map iframe { height: 520px; }
  .legend-column { max-height: none; }
}
.hint { color: var(--muted); font-size: 13px; }
.charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
}
.chart { margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfdff; }
.chart img { width: 100%; height: auto; }
.chart figcaption { color: var(--muted); font-size: 12px; margin-top: 6px; }
.legend-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.legend-grid .legend-card { max-width: none; }
.legend-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbfdff;
}
.legend { margin: 0; }
.legend-title { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.gradient { height: 10px; border-radius: 3px; }
.legend-range { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-top: 3px; }
.legend-rows { display: grid; grid-template-columns: 1fr; gap: 0; }
.legend-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: 12px; }
.legend-note { margin: 7px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.4; }
.swatch { width: 13px; height: 13px; display: inline-block; border-radius: 3px; border: 1px solid #999; flex: 0 0 auto; }
.table-wrap { overflow: auto; max-height: 620px; border: 1px solid var(--line); border-radius: 8px; }
.data-table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; }
.data-table th, .data-table td { border: 1px solid var(--line); padding: 5px 7px; text-align: left; white-space: nowrap; }
.data-table thead th { position: sticky; top: 0; background: #eaf2f8; z-index: 1; }
.data-table tbody tr:nth-child(even) { background: #f8fafc; }
input[type="search"] {
  width: 100%;
  max-width: 420px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 7px;
  margin: 8px 0 12px;
}
footer { color: var(--muted); text-align: center; padding: 18px; font-size: 12px; }
code { background: #eef2f6; padding: 1px 4px; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>$title</h1>
  <p>$subtitle</p>
  <p><span class="badge badge-estimate">оценка</span> — значение получено моделью или допущением, а не взято из исходных данных.</p>
</header>
<main>
  <section>
    <h2>Ключевые показатели</h2>
    <div class="kpi-grid">$kpi</div>
  </section>
  <section>
    <h2>Карта социальной инфраструктуры</h2>
    <p class="hint">Образование, здравоохранение, социальная сфера, культура и рекреация. Слои переключаются в правом верхнем углу карты; справа показаны легенды только включённых слоёв.</p>
    <div class="map-layout">
      <div class="map">$map_social</div>
      <aside class="legend-column">
        <h3>Легенда</h3>
        <p class="hint">Включите слой в правом верхнем углу карты — его легенда появится здесь.</p>
        <div class="legend-grid">$legends_social</div>
      </aside>
    </div>
  </section>
  <section>
    <h2>Карта торговли, общепита, туризма и услуг</h2>
    <p class="hint">Коммерческая и бытовая сфера: торговля, общепит, туризм, услуги и транспорт. Слои доступности по группам включены в переключатель карты.</p>
    <div class="map-layout">
      <div class="map">$map_consumer</div>
      <aside class="legend-column">
        <h3>Легенда</h3>
        <p class="hint">Включите слой в правом верхнем углу карты — его легенда появится здесь.</p>
        <div class="legend-grid">$legends_consumer</div>
      </aside>
    </div>
  </section>
  <section>
    <h2>Графики</h2>
    <div class="charts">$charts</div>
  </section>
  <section>
    <h2>Полная таблица показателей</h2>
    <p class="hint">Все $blocks_count кварталов и все показатели pipeline. Начните вводить номер квартала или значение, чтобы отфильтровать строки.</p>
    <input id="tableSearch" type="search" placeholder="Фильтр по таблице: квартал, землепользование, морфотип...">
    <div class="table-wrap">$table</div>
  </section>
  <section>
    <h2>Обеспеченность школами</h2>
    <h3>По кварталам</h3>
    <div class="table-wrap">$provision_table</div>
    <h3>Связи «спрос → ёмкость»</h3>
    <div class="table-wrap">$links_table</div>
  </section>
  <section>
    <h2>Обеспеченность детскими садами</h2>
    <h3>По кварталам</h3>
    <div class="table-wrap">$kindergarten_provision_table</div>
    <h3>Связи «спрос → ёмкость»</h3>
    <div class="table-wrap">$kindergarten_links_table</div>
  </section>
  <section>
    <h2>Морфотипы</h2>
    <h3>Кластеры Spacematrix</h3>
    <div class="table-wrap">$clusters_table</div>
  </section>
  <section>
    <h2>Доступность</h2>
    <h3>Сводка матрицы времени в пути</h3>
    <div class="table-wrap">$accessibility_table</div>
    <p class="hint">Полная матрица $matrix_size×$matrix_size сохранена в <code>results/accessibility_matrix_walk_min.csv</code>.</p>
  </section>
  <section>
    <h2>Школы и детские сады</h2>
    <h3>Школы</h3>
    <div class="table-wrap">$schools_table</div>
    <h3>Детские сады</h3>
    <div class="table-wrap">$kindergartens_table</div>
  </section>
  <section>
    <h2>Доступность сервисов по группам</h2>
    <p class="hint">Для каждого квартала рассчитано время до ближайшего объекта группы и число объектов в 15-минутной пешеходной зоне. Социальная и коммерческая сферы вынесены на отдельные карты.</p>
    <div class="table-wrap">$service_group_access</div>
    <h3>Объекты по группам</h3>
    <div class="table-wrap">$service_groups_table</div>
  </section>
  <section>
    <h2>Сервисы по типам</h2>
    <div class="table-wrap">$services_table</div>
  </section>
  <section>
    <h2>Полнота показателей</h2>
    <div class="table-wrap">$completeness</div>
  </section>
  <section>
    <h2>Выводы</h2>
    $conclusions
  </section>
  <section>
    <h2>Источники</h2>
    $sources
  </section>
  <section>
    <h2>Единицы измерения</h2>
    <div class="table-wrap">$units</div>
  </section>
  <section>
    <h2>Ограничения и допущения</h2>
    $limitations
  </section>
  <section>
    <h2>Проверка результата</h2>
    <div class="table-wrap">$validation</div>
  </section>
</main>
<footer>$footer</footer>
<script>
document.addEventListener("DOMContentLoaded", function () {
  var input = document.getElementById("tableSearch");
  if (input) {
    input.addEventListener("input", function () {
      var query = this.value.toLowerCase();
      document.querySelectorAll("#metricsTable tbody tr").forEach(function (row) {
        row.style.display = row.innerText.toLowerCase().indexOf(query) === -1 ? "none" : "";
      });
    });
  }

  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || data.type !== "legend-state") return;
    var active = data.active || [];
    document.querySelectorAll(".legend-card").forEach(function (card) {
      if (data.map && card.dataset.map !== data.map) return;
      card.style.display = active.indexOf(card.dataset.layer) !== -1 ? "" : "none";
    });
  });
});
</script>
</body>
</html>
"""
    )

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    page = template.safe_substitute(
        title="Ивангородское городское поселение: BlocksNet-дашборд",
        subtitle="Учебный расчёт по готовым векторным слоям. Две карты — социальная и коммерческая сфера, графики и таблицы в одном HTML-файле.",
        blocks_count=str(len(frame)),
        matrix_size=str(len(frame)),
        kpi=kpi_html,
        map_social=social_map_html,
        legends_social=social_legends_html,
        map_consumer=consumer_map_html,
        legends_consumer=consumer_legends_html,
        charts=charts_html,
        table=full_table,
        provision_table=provision_table,
        links_table=links_table,
        kindergarten_provision_table=kindergarten_provision_table,
        kindergarten_links_table=kindergarten_links_table,
        clusters_table=clusters_table,
        accessibility_table=accessibility_html,
        schools_table=schools_table_html,
        kindergartens_table=kindergartens_table_html,
        services_table=services_table_html,
        service_groups_table=service_groups_table_html,
        service_group_access=service_group_access_html,
        completeness=completeness_table,
        conclusions=conclusions_html,
        sources=sources_html,
        units=units_table,
        limitations=limitations_html,
        validation=validation_table,
        footer=f"Собрано {generated}. Подложка карты: OpenStreetMap, ODbL 1.0.",
    )
    OUTPUT_PATH.write_text(page, encoding="utf-8")
    print(f"Дашборд сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
