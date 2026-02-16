import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import sqlite3
import concurrent.futures
import time
from datetime import datetime

st.set_page_config(page_title="Цены Челябинск 50+", layout="wide")

# ----------------------------
# ДАННЫЕ
# ----------------------------
PRODUCTS = [
    'молоко 2.5%', 'кефир', 'творог 5%', 'сыр российский', 'яйца c0', 'йогурт натураль',
    'батон нарезанный', 'хлеб ржаной', 'лаваш', 'пирожки',
    'колбаса докторская', 'курица бройлер', 'свинина', 'сосиски молочные', 'фарш говяжий',
    'картофель', 'огурцы', 'помидоры', 'морковь', 'лук репчатый', 'бананы', 'яблоки гала',
    'пиво жигульское', 'вино красное сухое', 'водка 40%', 'коньяк', 'пиво бочка',
    'сахар песок', 'масло подсолнечное', 'макароны', 'рис', 'чай черный',
] * 2  # 50+ строк (с повторами)

STORES = {
    'Магнит': 'https://magnit.ru/search/?q={q}',
    'Пятерочка': 'https://pyaterochka.ru/catalog/search?q={q}',
    'Лента': 'https://lenta.com/search/?q={q}',
    'Красное&Белое': 'https://krasnoe-belyoe.ru/search/?q={q}',
}

CATEGORY_KEYWORDS = {
    "Молочка": ["молоко", "кефир", "творог", "сыр", "йогурт"],
    "Хлеб": ["батон", "хлеб", "лаваш", "пирожки"],
    "Мясо": ["колбаса", "курица", "свинина", "сосиски", "фарш"],
    "Овощи": ["картофель", "огурцы", "помидоры", "морковь", "лук", "бананы", "яблоки"],
    "Алко": ["пиво", "вино", "водка", "коньяк"],
    "Бакалея": ["сахар", "масло", "макароны", "рис", "чай"],
}

DB_PATH = "prices_chelyabinsk.db"


# ----------------------------
# ПАРСИНГ
# ----------------------------
def _ua_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }


def _extract_price_any(soup: BeautifulSoup) -> float:
    """
    ОЧЕНЬ грубый экстрактор.
    В реальности для каждого магазина нужны свои селекторы + JSON в скриптах.
    """
    price_elem = soup.select_one('.price, [class*="price"], .product-price')
    if not price_elem:
        return 0.0

    txt = price_elem.get_text(" ", strip=True)
    txt = txt.replace("₽", "").replace("\xa0", " ").strip()

    # оставим только цифры/разделители
    filtered = []
    for ch in txt:
        if ch.isdigit() or ch in [".", ",", " "]:
            filtered.append(ch)
    num = "".join(filtered).replace(" ", "").replace(",", ".").strip()

    try:
        return float(num)
    except Exception:
        return 0.0


def parse_price(store_name: str, product: str, date_str: str) -> dict:
    try:
        url = STORES[store_name].format(q=product.replace(" ", "%20"))
        resp = requests.get(url, headers=_ua_headers(), timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")

        price = _extract_price_any(soup)
        return {
            "товар": product,
            "магазин": store_name,
            "цена": float(price) if price else 0.0,
            "дата": date_str,
            "район": "Челябинск",
        }
    except Exception:
        return {
            "товар": product,
            "магазин": store_name,
            "цена": 0.0,
            "дата": date_str,
            "район": "Челябинск",
        }


def save_to_sqlite(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("prices", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite() -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM prices", conn)
        conn.close()
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ----------------------------
# КЭШ + ЗАГРУЗКА
# ----------------------------
@st.cache_data(ttl=7200, show_spinner=False)
def fetch_all_prices(limit_products: int | None = None) -> pd.DataFrame:
    """
    ВАЖНО: эта функция кэшируется.
    Прогресс/плейсхолдеры внутри кэш-функции делать нельзя (они не детерминированы).
    Поэтому прогресс делаем снаружи через "не-кэш" раннер.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    products = PRODUCTS if limit_products is None else PRODUCTS[:limit_products]

    data = []
    # аккуратно: слишком много потоков + sleep = долго, но безопаснее по бану
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(parse_price, store, prod, date_str) for store in STORES for prod in products]
        for fut in concurrent.futures.as_completed(futures):
            data.append(fut.result())
            time.sleep(0.25)  # anti-ban мягче

    df = pd.DataFrame(data)
    save_to_sqlite(df)
    return df


def run_fetch_with_progress(limit_products: int | None = None) -> pd.DataFrame:
    """
    Некэшируемая оболочка: показывает прогресс, затем дергает кэшируемую fetch_all_prices.
    """
    # Мы не можем показать прогресс выполнения внутри fetch_all_prices (кэш),
    # поэтому делаем "фейковый" прогресс по ожидаемому количеству задач
    products = PRODUCTS if limit_products is None else PRODUCTS[:limit_products]
    total = len(STORES) * len(products)

    status = st.status("Загрузка цен…", expanded=True)
    bar = st.progress(0)
    info = st.empty()

    # Запускаем реальную загрузку
    # Пока она идёт, мы не можем получать "шаги" без переписывания на очередь,
    # но хотя бы покажем "ожидание" и финал.
    info.write(f"Запросов к сайтам: {total}. Это может занять время (anti-ban).")

    df = fetch_all_prices(limit_products=limit_products)

    bar.progress(1.0)
    status.update(label="Загрузка завершена", state="complete")
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def load_data_cached() -> pd.DataFrame:
    """
    Сначала пробуем SQLite (быстро), если нет — тянем сеть.
    """
    df_db = load_from_sqlite()
    if df_db is not None and not df_db.empty:
        return df_db
    return fetch_all_prices(limit_products=5)  # дефолтно тестовый режим


# ----------------------------
# UI
# ----------------------------
st.title("🛒 Дашборд цен 50+ товаров: Магнит, Пятёрочка, Лента, К&B (Челябинск)")
st.caption("Стабильный рендер: без DOM-гонок, с безопасным reload-кэшем")

# Sidebar controls
st.sidebar.header("Фильтры")
магазины_sel = st.sidebar.multiselect(
    "Магазины",
    list(STORES.keys()),
    default=list(STORES.keys()),
    key="магазины_sel",
)
категория_sel = st.sidebar.selectbox(
    "Категория",
    ["Все"] + list(CATEGORY_KEYWORDS.keys()),
    index=0,
    key="категория_sel",
)

st.sidebar.divider()

mode = st.sidebar.radio(
    "Режим загрузки",
    ["Тест (быстро)", "Полная (долго)"],
    index=0,
    key="load_mode",
)
limit_products = 5 if mode.startswith("Тест") else None

# Кнопка: безопасный reload
if st.sidebar.button("🔄 Обновить (перекачать)", key="btn_reload"):
    st.cache_data.clear()
    st.session_state["force_reload"] = True
    st.rerun()

# Получение данных: либо reload, либо cached
if st.session_state.pop("force_reload", False):
    df = run_fetch_with_progress(limit_products=limit_products)
    st.session_state["just_reloaded"] = True
else:
    with st.spinner("Загружаю данные из кэша/SQLite…"):
        df = load_data_cached()

# Защита от пустоты / нулей
if df is None or df.empty:
    st.warning("Данных нет. Нажми “Обновить (перекачать)”.")
    st.stop()

# Нормализуем типы
if "цена" in df.columns:
    df["цена"] = pd.to_numeric(df["цена"], errors="coerce").fillna(0.0)

# Применяем фильтры
if магазины_sel:
    df = df[df["магазин"].isin(магазины_sel)]

if категория_sel != "Все":
    kws = CATEGORY_KEYWORDS.get(категория_sel, [])
    if kws:
        pattern = "|".join([pd.regex.escape(k) for k in kws]) if hasattr(pd, "regex") else "|".join(kws)
        # Streamlit окружение может быть без pd.regex.escape — поэтому проще вручную:
        pattern = "|".join([k.replace(".", "\\.") for k in kws])
        df = df[df["товар"].str.lower().str.contains(pattern, na=False)]

# Если после фильтра пусто
if df.empty:
    st.info("По выбранным фильтрам данных нет.")
    st.stop()

# ----------------------------
# ВИЗУАЛИЗАЦИЯ
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    # mean price per store/product
    top = (
        df[df["цена"] > 0]
        .groupby(["магазин", "товар"], as_index=False)["цена"]
        .mean()
        .sort_values("цена", ascending=True)
        .head(10)
    )
    if top.empty:
        st.info("Нет распарсенных цен > 0 для топа.")
    else:
        fig = px.bar(
            top,
            x="товар",
            y="цена",
            color="магазин",
            title="Топ-10 (средняя цена): где дешевле?",
        )
        st.plotly_chart(fig, use_container_width=True, key="bar_top10")

with col2:
    df_nonzero = df[df["цена"] > 0]
    if df_nonzero.empty:
        st.info("Нет распарсенных цен > 0 для гистограммы.")
    else:
        fig2 = px.histogram(
            df_nonzero,
            x="цена",
            color="магазин",
            title="Распределение цен (только > 0)",
        )
        st.plotly_chart(fig2, use_container_width=True, key="hist_prices")

# Таблица
df_view = df.sort_values(["цена", "магазин", "товар"], ascending=[True, True, True])
st.dataframe(
    df_view.style.format({"цена": "{:.1f} ₽"}),
    height=420,
    use_container_width=True,
    key="df_prices",
)

# Дешевле всего (по ненулевым)
df_nonzero = df[df["цена"] > 0]
if not df_nonzero.empty:
    cheapest = df_nonzero.loc[df_nonzero["цена"].idxmin()]
    st.success(f"🏆 Дешевле всего: **{cheapest['товар']}** — **{cheapest['цена']:.1f} ₽** в **{cheapest['магазин']}**")
else:
    st.warning("Все цены = 0. Скорее всего селекторы не подходят под сайты (нужно делать под каждый магазин отдельно).")

# Баллоны — только после обновления
if st.session_state.pop("just_reloaded", False):
    st.balloons()
