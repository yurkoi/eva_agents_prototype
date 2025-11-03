from pydantic import BaseModel, Field, validator
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from typing import Optional, List, Dict, Any, Tuple, Annotated

import sqlite3
import json
from datetime import datetime

import os
import re
from dotenv import load_dotenv

import chromadb
from openai import OpenAI
load_dotenv()


CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_data")
COLL_PREFIX      = os.getenv("CHROMA_COLLECTION_PREFIX", "eva_products_")
MODEL_EMB        = os.getenv("MODEL_EMB", "text-embedding-3-small")
VECTOR_NAME      = os.getenv("VECTOR_NAME", "title")
TOP_K_DEFAULT    = int(os.getenv("SEARCH_TOPK", "5"))

mcp = FastMCP("EVA_search_chroma")
oa = OpenAI()
chroma = chromadb.PersistentClient(path=CHROMA_PATH)

SHOP_DB_PATH = os.getenv("SHOP_DB_PATH", "shop.db")
def _odb():
    conn = sqlite3.connect(SHOP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
# ---------- Validation / helpers ----------
_UA_PHONE_RE = re.compile(r"^\+?38?0\d{9}$")  # дозволимо 380XXXXXXXXX або +380XXXXXXXXX або 0XXXXXXXXX

def _normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone: return None
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("80") and len(digits)==11:  # рідкі кейси
        digits = "380" + digits[2:]
    if digits.startswith("0") and len(digits)==10:
        digits = "38" + digits
    if not digits.startswith("38"):
        digits = "38" + digits  # нормалізуємо до 380...
    if not digits.startswith("380"):
        digits = "3" + digits   # страховка
    normalized = f"+{digits}"
    return normalized if _UA_PHONE_RE.match(digits) else phone  # якщо не UA-патерн — залишимо як є

def _gen_order_no(conn) -> str:
    today = datetime.utcnow().strftime("%Y%m%d")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM orders WHERE created_at LIKE ?", (datetime.utcnow().strftime("%Y-%m-%d")+"%",))
    seq = (cur.fetchone()[0] or 0) + 1
    return f"EVA-{today}-{seq:06d}"

# ---------- Models ----------
class OrderItem(BaseModel):
    sku: str = Field(..., description="Артикул")
    qty: int = Field(..., gt=0, description="Кількість")
    price: float = Field(..., ge=0, description="Ціна за одиницю")
    title: Optional[str] = Field(None, description="Назва (опціонально)")

class CustomerInfo(BaseModel):
    first_name: str
    last_name: str
    city: str
    np_branch: str = Field(..., description="Відділення/поштомат Нової пошти (№ або адреса)")
    phone: str
    note: Optional[str] = None

    @validator("first_name","last_name","city","np_branch", pre=True)
    def _clean_str(cls, v):
        return (v or "").strip() or v
    @validator("phone", pre=True)
    def _clean_phone(cls, v):
        return _normalize_phone(v)

# ---------- Tools ----------
@mcp.tool(
    name="order_place",
    description=(
        "Оформити замовлення. Передай customer_id, items=[{sku, qty, price, title?}] "
        "та дані клієнта {first_name, last_name, city, np_branch, phone, note?}. "
        "Повертає order_id, order_no, статус, total."
    )
)
def order_place(
    customer_id: str,
    items: List[OrderItem],
    customer: CustomerInfo,
) -> CallToolResult:
    try:
        if not items:
            return CallToolResult(content=[TextContent(type="text", text="Порожній список items")], isError=True)

        total = 0.0
        items_plain: List[Dict] = []
        for it in items:
            row = {"sku": it.sku, "qty": int(it.qty), "price": float(it.price)}
            if it.title: row["title"] = it.title
            total += row["price"] * row["qty"]
            items_plain.append(row)

        now = _now()
        conn = _odb(); cur = conn.cursor()

        order_no = _gen_order_no(conn)
        cur.execute("""
            INSERT INTO orders (
                order_no, customer_id, status, total, items_json,
                first_name, last_name, city, np_branch, phone, note,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            order_no, customer_id, "pending", total, json.dumps(items_plain, ensure_ascii=False),
            customer.first_name, customer.last_name, customer.city, customer.np_branch, customer.phone, customer.note,
            now, now
        ))
        oid = cur.lastrowid

        # status history
        cur.execute("""
            INSERT INTO order_status_history(order_id, old_status, new_status, changed_at, note)
            VALUES(?,?,?,?,?)
        """, (oid, None, "pending", now, "Order placed"))

        conn.commit(); conn.close()

        txt = json.dumps({
            "order_id": oid,
            "order_no": order_no,
            "status": "pending",
            "total": round(total, 2)
        }, ensure_ascii=False)
        return CallToolResult(content=[TextContent(type="text", text=txt)])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Помилка order_place: {e}")], isError=True)

@mcp.tool(
    name="order_update_status",
    description="Оновити статус замовлення: pending / confirmed / paid / shipped / delivered / cancelled. Пише історію."
)
def order_update_status(order_id: int, new_status: str, note: Optional[str] = None) -> CallToolResult:
    try:
        new_status = (new_status or "").strip().lower()
        allowed = {"pending","confirmed","paid","shipped","delivered","cancelled"}
        if new_status not in allowed:
            return CallToolResult(content=[TextContent(type="text", text=f"Недопустимий статус: {new_status}")], isError=True)

        conn = _odb(); cur = conn.cursor()
        cur.execute("SELECT status FROM orders WHERE id=?", (order_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return CallToolResult(content=[TextContent(type="text", text=f"Замовлення {order_id} не знайдено")], isError=True)

        old_status = row["status"]
        cur.execute("UPDATE orders SET status=?, updated_at=? WHERE id=?", (new_status, _now(), order_id))
        cur.execute("""
            INSERT INTO order_status_history(order_id, old_status, new_status, changed_at, note)
            VALUES(?,?,?,?,?)
        """, (order_id, old_status, new_status, _now(), note))
        conn.commit(); conn.close()

        return CallToolResult(content=[TextContent(type="text", text=f"order_id={order_id} status={new_status}")])
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Помилка order_update_status: {e}")], isError=True)


@mcp.tool(
    name="order_get",
    description="Отримати деталі замовлення за order_id або order_no."
)
def order_get(order_id: Optional[int] = None, order_no: Optional[str] = None) -> CallToolResult:
    try:
        if not order_id and not order_no:
            return CallToolResult(content=[TextContent(type="text", text="Потрібен order_id або order_no")], isError=True)
        conn = _odb(); cur = conn.cursor()
        if order_id:
            cur.execute("SELECT * FROM orders WHERE id=?", (order_id,))
        else:
            cur.execute("SELECT * FROM orders WHERE order_no=?", (order_no,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return CallToolResult(content=[TextContent(type="text", text="Замовлення не знайдено")], isError=True)

        cur.execute("SELECT old_status,new_status,changed_at,note FROM order_status_history WHERE order_id=? ORDER BY id", (row["id"],))
        hist = [dict(r) for r in cur.fetchall()]
        conn.close()

        data = dict(row)
        data["items"] = json.loads(data.pop("items_json") or "[]")
        data["history"] = hist
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))])
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Помилка order_get: {e}")], isError=True)


@mcp.tool(
    name="order_list_by_customer",
    description="Список замовлень користувача з пагінацією (limit/offset)."
)
def order_list_by_customer(customer_id: str, limit: int = 20, offset: int = 0) -> CallToolResult:
    try:
        conn = _odb(); cur = conn.cursor()
        cur.execute("""
            SELECT id, order_no, status, total, created_at
            FROM orders WHERE customer_id=?
            ORDER BY id DESC LIMIT ? OFFSET ?
        """, (customer_id, int(limit), int(offset)))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(rows, ensure_ascii=False, indent=2))])
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Помилка order_list_by_customer: {e}")], isError=True)


@mcp.tool(
    name="order_add_comment",
    description="Додати службовий коментар до історії замовлення (не змінює статус)."
)
def order_add_comment(order_id: int, note: str) -> CallToolResult:
    try:
        conn = _odb(); cur = conn.cursor()
        cur.execute("SELECT id FROM orders WHERE id=?", (order_id,))
        if not cur.fetchone():
            conn.close()
            return CallToolResult(content=[TextContent(type="text", text="Замовлення не знайдено")], isError=True)
        cur.execute("""
            INSERT INTO order_status_history(order_id, old_status, new_status, changed_at, note)
            VALUES(?,?,?,?,?)
        """, (order_id, None, None, _now(), note))
        conn.commit(); conn.close()
        return CallToolResult(content=[TextContent(type="text", text="OK")])
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Помилка order_add_comment: {e}")], isError=True)



def _get_collection(prefix=COLL_PREFIX, vector_name=VECTOR_NAME):
    return chroma.get_or_create_collection(name=f"{prefix}{vector_name}")

def _embed(text: str) -> list[float]:
    return oa.embeddings.create(model=MODEL_EMB, input=[text]).data[0].embedding


@mcp.tool(
    name="product_search",
    description="Глобальний семантичний пошук по товарах EVA. Опційно: min_price/max_price — діапазон цін (грн).")
def product_search(
    query: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> CallToolResult:
    try:
        col  = _get_collection()
        qvec = _embed(query)

        where = None
        try:
            lo = float(min_price) if min_price is not None else None
            hi = float(max_price) if max_price is not None else None
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo  # авто-_swap, якщо переплутали місцями

            if lo is not None or hi is not None:
                rng = {}
                if lo is not None:
                    rng["$gte"] = lo
                if hi is not None:
                    rng["$lte"] = hi
                where = {
                    "$or": [
                        {"spec_price": rng},
                        {"price": rng},
                    ]
                }
        except (TypeError, ValueError):
            where = None  # некоректні числа — просто ігноруємо фільтр

        res = col.query(
            query_embeddings=[qvec],
            n_results=18,
            include=["metadatas", "distances"],
            where=where)

        ids       = (res.get("ids") or [[]])[0]
        metas     = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        if not ids:
            txt = "Нічого не знайдено."
        else:
            lines = []
            if min_price is not None or max_price is not None:
                tag_lo = f"{float(min_price):.0f}" if min_price is not None else "—"
                tag_hi = f"{float(max_price):.0f}" if max_price is not None else "—"
                lines.append(f"[price_range] {tag_lo} … {tag_hi} грн")

            for i, rid in enumerate(ids, start=1):
                meta = metas[i-1] if i-1 < len(metas) else {}
                d    = distances[i-1] if i-1 < len(distances) else None
                sim  = (1.0 - float(d)) if d is not None else None
                sim_s = f"{sim:.3f}" if sim is not None else "—"

                lines.append(f"\n#{i} id={rid} similarity={sim_s}")
                for k in sorted(meta.keys()):
                    v = meta[k]
                    if isinstance(v, float):
                        v = f"{v:.4f}".rstrip("0").rstrip(".")
                    lines.append(f"  {k}: {v}")

            txt = "\n".join(lines)

        return CallToolResult(content=[TextContent(type="text", text=txt)])

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Помилка пошуку: {e}")],
            isError=True,
            isRetryable=False,
        )
        
@mcp.tool(
    name="faq_search",
    description="Семантичний пошук по найчастішим запитанням FAQ. Повертає топ релевантних блоків."
)
def faq_search(
    query: str,
    top_k: int = 3,
) -> CallToolResult:
    try:
        col  = chroma.get_or_create_collection(name=f"faq_eva")
        qvec = _embed(query)

        res = col.query(
            query_embeddings=[qvec],
            n_results=int(top_k),
            include=["metadatas", "documents", "distances"],
        )

        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        if not docs:
            return CallToolResult(content=[TextContent(type="text", text="Нічого не знайдено.")])

        lines = ["[faq_search results]"]
        for i, doc in enumerate(docs, start=1):
            meta = metas[i-1] if i-1 < len(metas) else {}
            d    = dists[i-1] if i-1 < len(dists) else None
            sim  = (1.0 - float(d)) if d is not None else None
            sim_s = f"{sim:.3f}" if sim is not None else "—"

            title = meta.get("title") or (doc.split("\n", 1)[0].strip() if doc else "")
            preview = (doc or "").strip()

            lines.append(f"\n#{i} similarity={sim_s}")
            if title:
                lines.append(f"  title: {title}")
            q = meta.get("question")
            a = meta.get("answer")
            if q or a:
                if q: lines.append(f"  Q: {q}")
                if a: 
                    short_a = a if len(a) <= 700 else a[:700].rstrip() + "…"
                    lines.append("  A: " + short_a.replace("\n", "\n     "))
            else:
                lines.append("  text:")
                lines.append("  " + preview.replace("\n", "\n  "))

        txt = "\n".join(lines)
        return CallToolResult(content=[TextContent(type="text", text=txt)])

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Помилка пошуку: {e}")],
            isError=True,
            isRetryable=False,
        )
        
        

GENDER_MAP = {
    "ж": "Жіноча", "жіноча": "Жіноча", "женская": "Жіноча", "female": "Жіноча",
    "ч": "Чоловіча", "чоловіча": "Чоловіча", "мужская": "Чоловіча", "male": "Чоловіча",
    "у": "Унісекс", "унісекс": "Унісекс", "unisex": "Унісекс",
}

def _norm_gender(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    key = x.strip().lower()
    return GENDER_MAP.get(key, x)

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _flatten(res: Dict[str, Any], key: str) -> List[Any]:
    arr = res.get(key) or []
    if arr and isinstance(arr[0], list):
        return arr[0]
    return arr

def _build_where(
    *,
    gender: Optional[str],
    brand: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
    country: Optional[str],
    series: Optional[str],
    sku_in: Optional[List[str]]
) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []

    if gender:
        clauses.append({"gender": _norm_gender(gender)})
    if brand:
        clauses.append({"brand": {"$eq": brand}})
    if country:
        clauses.append({"country": {"$eq": country}})
    if series:
        clauses.append({"series": {"$eq": series}})

    if min_price is not None:
        clauses.append({"price": {"$gte": _safe_float(min_price)}})
    if max_price is not None:
        clauses.append({"price": {"$lte": _safe_float(max_price)}})

    if sku_in:
        # Chroma поддерживает $in по метаданным
        clauses.append({"sku": {"$in": list({str(s) for s in sku_in})}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

@mcp.tool(
    name="perfume_search",
    description="Семантичний пошук по Chroma-колекції парфумів. Повертає JSON (items, total) + превʼю."
)
def perfume_search(
    query: str = "парфуми",
    limit: int = 7,
    offset: int = 0,
    gender: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    country: Optional[str] = None,
    series: Optional[str] = None,
    sku_in: Optional[List[str]] = None,
) -> CallToolResult:
    """
    Возвращает:
    {
      "source": "chroma:eva_perfumes",
      "total": int,                  # оценка числа канд. (по n_results до offset+limit)
      "items": [
        {
          "doc": str,                # твой конкатенированный документ
          "meta": {...},             # метаданные (brand, name, gender, price, url, sku, ...)
          "distance": float | null   # cosine distance, если доступно
        }, ...
      ]
    }
    Плюс человекочитаемый preview ниже JSON для удобного логирования.
    """
    try:
        
        col = chroma.get_or_create_collection(name="eva_perfumes")
        qvec = _embed(query or "парфуми")

        need = max(0, limit) + max(0, offset)
        need = max(need, 1)

        where = _build_where(
            gender=gender,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            country=country,
            series=series,
            sku_in=sku_in,
        )

        res = col.query(
            query_embeddings=[qvec],
            n_results=need,
            include=["documents", "metadatas", "distances"],
            where=where
        )

        docs = _flatten(res, "documents")
        metas = _flatten(res, "metadatas")
        dists = _flatten(res, "distances")

        # нормализуем длины массивов
        L = max(len(docs), len(metas), len(dists))
        def _get(arr, i, default=None):
            try:
                return arr[i]
            except Exception:
                return default

        raw_items: List[Tuple[str, Dict[str, Any], Any]] = []
        for i in range(L):
            doc  = _get(docs, i, "")
            meta = _get(metas, i, {}) or {}
            dist = _get(dists, i, None)
            if not isinstance(doc, str):
                doc = "" if doc is None else str(doc)
            if not isinstance(meta, dict):
                try:
                    meta = dict(meta)
                except Exception:
                    meta = {}
            raw_items.append((doc, meta, dist))

        # пагинация (offset/limit)
        sliced = raw_items[offset: offset + limit] if raw_items else []

        items = [
            {
                "doc": doc,
                "meta": meta,
                "distance": float(dist) if dist is not None else None
            }
            for (doc, meta, dist) in sliced
        ]

        payload = {
            "source": "chroma:eva_perfumes",
            "total": len(raw_items),
            "items": items,
        }

        # человекочитаемый превью (удобно логировать/читать глазами)
        if not items:
            preview = "Нічого не знайдено."
        else:
            lines = []
            for it in items:
                m = it["meta"] or {}
                name = m.get("name") or m.get("brand") or "—"
                price = m.get("price")
                g = m.get("gender")
                sku = m.get("sku")
                url = m.get("url")
                dist = it.get("distance")
                lines.append(
                    f"{name} | {g or '—'} | {price if price is not None else '—'} грн "
                    f"| sku={sku or '—'} | dist={dist if dist is not None else '—'}"
                )
                if url:
                    lines.append(f"  {url}")
            preview = "\n".join(lines)

        txt = json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n" + preview

        return CallToolResult(
            content=[TextContent(type="text", text=txt)]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Помилка пошуку: {e}")],
            isError=True,
            isRetryable=False,
        )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
