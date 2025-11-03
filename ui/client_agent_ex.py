import os
import re
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, OpenAIConversationsSession
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from openai.types.responses import ResponseTextDeltaEvent

from promtps import (
    ORDER_CONSULTANT,
    PARFUMER_PROMPT,
    KC_CONSULTANT,
    ROUTER_SYSTEM_PROMPT,
)

load_dotenv()

AGENT_LABELS: Dict[str, str] = {
    "Consultant_Perfumer": "[Парфумер консультант підключено]",
    "Consultant_FAQ": "[FAQ консультант підключено]",
    "Product_Orders_Consultant": "[Консультант замовлень підключено]",
}

VALID_LABELS = ("FAQ", "ORDER", "PERFUME", "SMALLTALK")
ROUTE_TOKEN_RE = re.compile(r"\b(FAQ|ORDER|PERFUME|SMALLTALK)\b", re.IGNORECASE)


IMG_LINE_RE = re.compile(
    r"(?mi)^(?!\s*!\[\]\()\s*Зображення\s*:\s*(https?://\S+\.(?:png|jpe?g|webp|gif)(?:\?\S+)?)\s*$"
)
def inline_image_markdown(md: str) -> str:
    return IMG_LINE_RE.sub(r"![](\1)", md)


def build_router_user_msg(user_text: str) -> str:
    return (
        "—\n"
        "Повідомлення користувача:\n"
        f"{user_text}\n"
        "—\n"
        "Відповідай ОДНИМ СЛОВОМ з множини {FAQ|ORDER|PERFUME|SMALLTALK}. Без пояснень."
    )

async def ask_router(router_agent: Agent, user_text: str) -> str:
    """Run RouterAgent and return single class token."""
    router_user_msg = build_router_user_msg(user_text)
    result = Runner.run_streamed(router_agent, router_user_msg)

    buffer: List[str] = []
    async for ev in result.stream_events():
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            buffer.append(ev.data.delta or "")

    raw = "".join(buffer).strip()
    m = ROUTE_TOKEN_RE.search(raw.upper())
    label = m.group(1).upper() if m else None
    return label if label in VALID_LABELS else "FAQ"


async def stream_agent_to_placeholder(agent: Agent, user_msg: str, session: OpenAIConversationsSession, container) -> str:
    top = container.empty()
    result = Runner.run_streamed(agent, user_msg, session=session)
    out_chunks: List[str] = []

    async for ev in result.stream_events():
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            out_chunks.append(ev.data.delta or "")
            cur = "".join(out_chunks)
            top.markdown(inline_image_markdown(cur))  # <-- інлайн-картинки

    return "".join(out_chunks)


@asynccontextmanager
async def mcp_server(url: str):
    async with MCPServerStreamableHttp(
        name="EVA MCP Server",
        params={"url": url},
        cache_tools_list=True,
    ) as server:
        yield server

st.set_page_config(page_title="EVA Consult 🌸", page_icon="!", layout="wide")

with st.sidebar:
    st.header("⚙️ Налаштування")
    mcp_url = st.text_input("MCP URL", os.getenv("MCP_URL", "http://localhost:8000/mcp"))

    st.caption("Маршрутизація")
    override = st.selectbox("Manual override (optional)", ["auto", "FAQ", "PERFUME", "ORDER"], index=0)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Очистити чат", use_container_width=True):
            st.session_state.clear()
            # Явно скинемо маркери маршруту (на випадок часткового reset’у)
            st.session_state.last_route = None
            st.session_state.last_agent = None
            st.rerun()
    with col_b:
        st.write(" ")

    st.markdown("**Підказки**\n- Вкажіть валідний MCP endpoint.\n- Для діагностики увімкніть manual override.")

st.title("🌸 EVA консультації")
st.caption("Команди: manual override в сайдбарі. Авто-маршрутизація за замовчуванням.")

# Session state
if "base_session" not in st.session_state:
    st.session_state.base_session = OpenAIConversationsSession()
if "history" not in st.session_state:
    st.session_state.history = []

if "last_route" not in st.session_state:
    st.session_state.last_route = None
if "last_agent" not in st.session_state:
    st.session_state.last_agent = None

# ---------- rendering helpers ----------
ROUTE_BADGE = {
    "FAQ": "🟦 FAQ",
    "ORDER": "🟧 ORDER",
    "PERFUME": "🟩 PERFUME",
    "SMALLTALK": "🟨 SMALLTALK",
}

def render_route_badge(route: str, agent_name: str) -> str:
    badge = ROUTE_BADGE.get(route, route)
    agent_label = AGENT_LABELS.get(agent_name, "")
    return f"**{badge}** — {agent_label}"

for msg in st.session_state.history:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif role == "route":
        with st.chat_message("assistant"):
            st.info(msg["content"]) 

user_msg = st.chat_input("Введіть повідомлення…")

async def handle_message(user_text: str):
    router_agent = Agent(
        name="Triage_Agent",
        instructions=ROUTER_SYSTEM_PROMPT,
    )

    if override != "auto":
        route = override
    else:
        try:
            route = await ask_router(router_agent=router_agent, user_text=user_text)
        except Exception as e:
            route = "FAQ"  # fallback
            st.toast(f"Помилка маршрутизації: {e}")

    async with mcp_server(mcp_url) as server:
        perfumer_agent = Agent(
            name="Consultant_Perfumer",
            instructions=PARFUMER_PROMPT,
            mcp_servers=[server],
            # model_settings=common_settings,
        )
        faq_agent = Agent(
            name="Consultant_FAQ",
            instructions=KC_CONSULTANT,
            mcp_servers=[server],
            # model_settings=common_settings,
        )
        order_agent = Agent(
            name="Product_Orders_Consultant",
            instructions=ORDER_CONSULTANT,
            mcp_servers=[server],
            # model_settings=common_settings,
        )

        agents_by_label = {
            "PERFUME": perfumer_agent,
            "FAQ": faq_agent,
            "ORDER": order_agent,
            "SMALLTALK": faq_agent,
        }

        agent = agents_by_label.get(route, faq_agent)
        route_msg = render_route_badge(route, agent.name)
        route_changed = (
            route != st.session_state.last_route
            or agent.name != st.session_state.last_agent
        )

        if route_changed:
            st.session_state.history.append({"role": "route", "content": route_msg})
            with st.chat_message("assistant"):
                st.info(route_msg)
            st.session_state.last_route = route
            st.session_state.last_agent = agent.name

        with st.chat_message("assistant"):
            msg_box = st.container()
            final_text = await stream_agent_to_placeholder(
                agent, user_text, st.session_state.base_session, msg_box
            )

    st.session_state.history.append({"role": "assistant", "content": final_text or ""})

if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        asyncio.run(handle_message(user_msg))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(handle_message(user_msg))
