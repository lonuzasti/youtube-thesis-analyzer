import streamlit as st
from googleapiclient.discovery import build
from google import genai
from google.genai import types
from langdetect import detect
from textblob import TextBlob
from transformers import pipeline
from janome.tokenizer import Tokenizer
import plotly.express as px
import pandas as pd
import numpy as np
import collections
import re
import os
import json
import time

# 画面の設定
st.set_page_config(page_title="YouTube 日米コメント感情・モデル比較ダッシュボード", layout="wide")
st.title("YouTube 日英コメント 感情・モデル比較ダッシュボード")

# 履歴保存ディレクトリ
HISTORY_DIR = "analysis_history"
MAX_HISTORY_COUNT = 10
os.makedirs(HISTORY_DIR, exist_ok=True)

# 日本語BERTモデルの読み込み
@st.cache_resource
def load_ja_model():
    return pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")

ja_sentiment_analyzer = load_ja_model()
tokenizer = Tokenizer()

# APIキーの自動読み込み処理
def load_secret(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    for path in [".streamlit/secrets.toml", "secrets.toml"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if key_name in line and "=" in line:
                            return line.split("=")[1].strip().replace('"', '').replace("'", "")
            except Exception:
                pass
    return ""

default_yt_key = load_secret("YOUTUBE_API_KEY")
default_gemini_key = load_secret("GEMINI_API_KEY")

# 履歴管理用ユーティリティ
def get_saved_histories():
    histories = []
    if os.path.exists(HISTORY_DIR):
        files = sorted(
            [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, x)),
            reverse=True
        )
        for f in files:
            try:
                with open(os.path.join(HISTORY_DIR, f), "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                    histories.append(meta)
            except Exception:
                pass
    return histories

def save_to_history(video_id, video_title, df_en, df_ja):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{video_id}"
    meta_path = os.path.join(HISTORY_DIR, f"{base_name}.json")
    csv_path = os.path.join(HISTORY_DIR, f"{base_name}.csv")
    
    combined_df = pd.concat([df_en, df_ja], ignore_index=True)
    combined_df.to_csv(csv_path, index=False, encoding="utf-8_sig")
    
    meta = {
        "id": base_name,
        "video_id": video_id,
        "video_title": video_title,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "en_count": len(df_en),
        "ja_count": len(df_ja),
        "csv_file": f"{base_name}.csv"
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
        
    histories = get_saved_histories()
    if len(histories) > MAX_HISTORY_COUNT:
        for old in histories[MAX_HISTORY_COUNT:]:
            old_id = old["id"]
            for ext in [".json", ".csv"]:
                p = os.path.join(HISTORY_DIR, f"{old_id}{ext}")
                if os.path.exists(p):
                    os.remove(p)

def load_history_data(history_item):
    csv_path = os.path.join(HISTORY_DIR, history_item["csv_file"])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8_sig")
        df_en = df[df["lang"] == "en"].copy()
        df_ja = df[df["lang"] == "ja"].copy()
        if "is_match" in df_ja.columns:
            df_ja["is_match"] = df_ja["is_match"].astype(bool)
        return df_en, df_ja
    return pd.DataFrame(), pd.DataFrame()

# サイドバー設定
st.sidebar.header("⚙️ 設定")
api_key = st.sidebar.text_input("YouTube API Key", value=default_yt_key, type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", value=default_gemini_key, type="password")
max_comments_to_fetch = st.sidebar.slider("取得する最大コメント数", min_value=100, max_value=500, value=300, step=100)

# 履歴読み込みUI
st.sidebar.markdown("---")
st.sidebar.header("🕒 過去の分析履歴")
saved_histories = get_saved_histories()

if saved_histories:
    options = ["-- 履歴から選択して再表示 --"] + [f"{h['date']} | {h['video_title'][:20]}..." for h in saved_histories]
    selected_idx = st.sidebar.selectbox("過去の分析を再表示:", range(len(options)), format_func=lambda x: options[x])
    
    if selected_idx > 0:
        target_history = saved_histories[selected_idx - 1]
        if st.sidebar.button("この履歴を読み込む"):
            df_en_hist, df_ja_hist = load_history_data(target_history)
            st.session_state.df_en = df_en_hist
            st.session_state.df_ja = df_ja_hist
            st.session_state.video_id = target_history["video_id"]
            st.session_state.video_title = target_history["video_title"]
            st.sidebar.success("履歴を読み込みました！")
            
    if st.sidebar.button("🗑️ 履歴をすべて消去"):
        for f in os.listdir(HISTORY_DIR):
            os.remove(os.path.join(HISTORY_DIR, f))
        st.sidebar.success("履歴を消去しました。")
        st.rerun()
else:
    st.sidebar.caption("保存された履歴はありません（分析すると自動保存されます）。")

# セッション状態の初期化
if "df_en" not in st.session_state:
    st.session_state.df_en = pd.DataFrame()
if "df_ja" not in st.session_state:
    st.session_state.df_ja = pd.DataFrame()
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "video_title" not in st.session_state:
    st.session_state.video_title = ""

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_video_title(api_key, video_id):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        items = response.get("items", [])
        if items:
            return items[0]["snippet"]["title"]
    except Exception:
        pass
    return "タイトル取得不可"

def get_comments(api_key, video_id, max_count=300):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    
    while len(comments) < max_count:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_count - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            break
            
        for item in items:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            likes = int(item["snippet"]["topLevelComment"]["snippet"]["likeCount"])
            comments.append({"comment": comment, "likes": likes})
            
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
            
    return pd.DataFrame(comments)

# ハイブリッド補正 (BERT用)
def apply_hybrid_ja_correction(text, polarity, category):
    pos_patterns = [
        r"[😂🤣😭🥺❤️✨👏🥰😍😻👍💯🎉]",
        r"(笑+|草+|www+|ｗｗｗ+|ワロタ|わろた)",
        r"(好き|最高|神|尊い|かわいい|可愛い|おもろい|面白い|癒やし|癒し|愛おしい|感謝|嬉しい)"
    ]
    strong_neg_patterns = [
        r"(嫌い|最悪|クソ|ゴミ|不快|キモい|きもい|炎上|通報|害悪|うざい|ウザい)"
    ]
    has_pos_signal = any(re.search(pat, text) for pat in pos_patterns)
    has_strong_neg = any(re.search(pat, text) for pat in strong_neg_patterns)
    
    if has_pos_signal and not has_strong_neg:
        if category == "Negative (否定的)":
            polarity = 0.7
            category = "Positive (肯定的)"
        elif category == "Neutral (中立)":
            polarity = 0.5
            category = "Positive (肯定的)"
            
    return polarity, category

# 日本語BERT判定関数
def analyze_ja_batch_bert(texts, batch_size=32):
    truncated_texts = [str(t)[:128] if str(t).strip() else " " for t in texts]
    results = []
    
    for i in range(0, len(truncated_texts), batch_size):
        batch = truncated_texts[i:i + batch_size]
        res_list = ja_sentiment_analyzer(batch)
        
        for text_orig, res in zip(texts[i:i + batch_size], res_list):
            label = res["label"]
            score = res["score"]
            if "POS" in label.upper():
                polarity = score
                category = "Positive (肯定的)"
            else:
                polarity = -score
                category = "Negative (否定的)"
                
            if score < 0.65:
                category = "Neutral (中立)"
                polarity = 0.0
                
            polarity, category = apply_hybrid_ja_correction(str(text_orig), polarity, category)
            results.append({"bert_sentiment": polarity, "bert_category": category})
            
    return pd.DataFrame(results)

# Gemini 3.6 Flashによる文脈考慮判定（レートリミット・混雑完全防御版）
def analyze_with_gemini(texts, video_title, gemini_key, batch_size=100):
    client = genai.Client(api_key=gemini_key)
    all_results = []
    
    prompt_template = """
あなたはYouTubeコメントの高度な感情・世論分析を行う専門家です。
以下の動画タイトル（背景文脈）を踏まえ、各コメントの投稿者に対するスタンス・感情を判定してください。

【重要な判定基準】
- 動画タイトルが悩み・愚痴・重大発表（例: 限界、辛い、結婚、移籍、引退、休止）であっても、コメントが共感・励まし・労い・祝福である場合は「Positive (肯定的)」と分類してください。
- 皮肉や当てこすりは文脈を踏まえて「Negative (否定的)」と分類してください。

【動画タイトル】: {title}

【コメント一覧】:
{comments_json}

以下のJSON配列形式のみで出力してください:
[
  {{"index": 0, "category": "Positive (肯定的)" または "Negative (否定的)" または "Neutral (中立)", "sentiment": -1.0から1.0の数値}},
  ...
]
"""
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        indexed_comments = [{"index": idx, "text": str(t)[:120]} for idx, t in enumerate(batch)]
        comments_json = json.dumps(indexed_comments, ensure_ascii=False)
        prompt = prompt_template.format(title=video_title, comments_json=comments_json)
        
        success = False
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                parsed = json.loads(response.text)
                parsed_dict = {item["index"]: item for item in parsed}
                
                for idx in range(len(batch)):
                    if idx in parsed_dict:
                        item = parsed_dict[idx]
                        cat = item.get("category", "Neutral (中立)")
                        if "Positive" in cat or "肯定" in cat:
                            category = "Positive (肯定的)"
                        elif "Negative" in cat or "否定" in cat:
                            category = "Negative (否定的)"
                        else:
                            category = "Neutral (中立)"
                        polarity = float(item.get("sentiment", 0.0))
                        all_results.append({"gemini_sentiment": polarity, "gemini_category": category})
                    else:
                        all_results.append({"gemini_sentiment": 0.0, "gemini_category": "Neutral (中立)"})
                
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(15 * (attempt + 1))
                else:
                    time.sleep(5)
                    
                if attempt == 3:
                    st.warning(f"一部バッチの処理でエラーが発生しました（中立補完）: {e}")
                    for _ in batch:
                        all_results.append({"gemini_sentiment": 0.0, "gemini_category": "Neutral (中立)"})
        
        if i + batch_size < len(texts):
            time.sleep(2.5)
                
    return pd.DataFrame(all_results)

def calculate_weighted_sentiment(scores, likes):
    if len(scores) == 0:
        return 0.0
    weights = likes + 1
    return np.average(scores, weights=weights)

def get_top_words_en(texts, top_n=10):
    words = []
    stop_words = set(["the", "a", "an", "is", "and", "to", "in", "it", "of", "for", "this", "that", "i", "you", "my", "with", "on", "was", "are", "but", "so", "be"])
    for text in texts:
        clean_text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
        tokens = clean_text.split()
        words.extend([w for w in tokens if len(w) > 2 and w not in stop_words])
    counter = collections.Counter(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["単語", "出現回数"])

def get_top_words_ja(texts, top_n=10):
    words = []
    stop_words = set(["こと", "もの", "これ", "それ", "よう", "そう", "さん", "ちゃん", "ため", "動画", "投稿", "自分", "本当"])
    for text in texts:
        tokens = tokenizer.tokenize(str(text))
        for token in tokens:
            part = token.part_of_speech.split(',')[0]
            word = token.surface
            if part == '名詞' and len(word) > 1 and word not in stop_words:
                words.append(word)
    counter = collections.Counter(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["単語", "出現回数"])

# メイン画面
url_input = st.text_input("分析したいYouTube動画のURLを入力してください")

if st.button("分析を開始する") and url_input and api_key:
    video_id = extract_video_id(url_input)
    
    if not video_id:
        st.error("有効なYouTube URLを入力してください。")
    elif not gemini_api_key:
        st.error("Gemini APIキーを入力してください。")
    else:
        with st.spinner(f"動画情報とコメントを取得し、BERTとGeminiで分析中..."):
            video_title = get_video_title(api_key, video_id)
            df = get_comments(api_key, video_id, max_count=max_comments_to_fetch)
            
            if df.empty:
                st.warning("コメントが取得できませんでした。")
            else:
                def check_lang(text):
                    try:
                        return detect(text)
                    except:
                        return "unknown"
                
                df["lang"] = df["comment"].apply(check_lang)
                df_en = df[df["lang"] == "en"].copy()
                df_ja = df[df["lang"] == "ja"].copy()
                
                # 英語の分析 (Gemini)
                if not df_en.empty:
                    res_en = analyze_with_gemini(df_en["comment"].tolist(), video_title, gemini_api_key)
                    df_en["gemini_sentiment"] = res_en["gemini_sentiment"].values
                    df_en["gemini_category"] = res_en["gemini_category"].values
                
                # 日本語の分析 (BERT と Gemini を同時実行)
                if not df_ja.empty:
                    res_bert = analyze_ja_batch_bert(df_ja["comment"].tolist())
                    df_ja["bert_sentiment"] = res_bert["bert_sentiment"].values
                    df_ja["bert_category"] = res_bert["bert_category"].values
                    
                    res_gemini = analyze_with_gemini(df_ja["comment"].tolist(), video_title, gemini_api_key)
                    df_ja["gemini_sentiment"] = res_gemini["gemini_sentiment"].values
                    df_ja["gemini_category"] = res_gemini["gemini_category"].values
                    
                    df_ja["is_match"] = df_ja["bert_category"] == df_ja["gemini_category"]
                
                st.session_state.df_en = df_en
                st.session_state.df_ja = df_ja
                st.session_state.video_id = video_id
                st.session_state.video_title = video_title
                save_to_history(video_id, video_title, df_en, df_ja)

# 結果表示エリア
if not st.session_state.df_en.empty or not st.session_state.df_ja.empty:
    df_en = st.session_state.df_en
    df_ja = st.session_state.df_ja
    video_id = st.session_state.video_id
    video_title = st.session_state.video_title
    
    st.info(f"🎬 **対象動画**: {video_title} (ID: `{video_id}`)")
    st.success(f"取得・分析完了: 英語 {len(df_en)} 件 / 日本語 {len(df_ja)} 件")
    
    combined_df = pd.concat([df_en, df_ja])
    csv_data = combined_df.to_csv(index=False).encode('utf-8_sig')
    st.download_button(
        label="📥 分析結果（BERT & Gemini比較データ）をCSVダウンロード",
        data=csv_data,
        file_name=f"youtube_analysis_comparison_{video_id}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🌐 日英感情分析・世論比較", "🔬 BERT vs Gemini 判定差分ビューア"])
    
    color_map = {
        "Positive (肯定的)": "#2b83ba",
        "Neutral (中立)": "#abdda4",
        "Negative (否定的)": "#d7191c"
    }

    # ==========================================
    # TAB 1: 日米感情分析・世論比較
    # ==========================================
    with tab1:
        col1, col2 = st.columns(2)
        
        # 🇺🇸 英語
        with col1:
            st.header("英語コメント分析 (Gemini推論)")
            if not df_en.empty:
                avg_en = df_en["gemini_sentiment"].mean()
                weighted_en = calculate_weighted_sentiment(df_en["gemini_sentiment"], df_en["likes"])
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("単純平均スコア", f"{avg_en:.2f}")
                with m2:
                    st.metric("世論支持スコア (加重平均)", f"{weighted_en:.2f}", delta=f"{weighted_en - avg_en:+.2f}")
                
                fig_pie_en = px.pie(
                    df_en, names="gemini_category", title="感情の割合 (英語)",
                    color="gemini_category", color_discrete_map=color_map, hole=0.3
                )
                st.plotly_chart(fig_pie_en, use_container_width=True)
                
                fig_hist_en = px.histogram(
                    df_en, x="gemini_sentiment", nbins=20,
                    title="📊 感情スコア分布 (英語: -1.0〜+1.0)",
                    labels={"gemini_sentiment": "感情スコア"},
                    color_discrete_sequence=["#2b83ba"]
                )
                st.plotly_chart(fig_hist_en, use_container_width=True)
                
                st.subheader("🔤 よく使われている英単語 Top 10")
                top_words_en = get_top_words_en(df_en["comment"])
                if not top_words_en.empty:
                    fig_bar_en = px.bar(top_words_en, x="出現回数", y="単語", orientation="h", text="出現回数")
                    fig_bar_en.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_bar_en, use_container_width=True)
                
                st.subheader("💬 英語コメント一覧・絞り込み")
                filter_en = st.radio(
                    "表示する感情を選択 (英語):",
                    ["すべて表示", "Positive (肯定的)", "Neutral (中立)", "Negative (否定的)"],
                    key="filter_en_tab1",
                    horizontal=True
                )
                if filter_en == "すべて表示":
                    display_en = df_en.sort_values(by="likes", ascending=False)
                else:
                    display_en = df_en[df_en["gemini_category"] == filter_en].sort_values(by="likes", ascending=False)
                    
                st.caption(f"表示中: {len(display_en)} 件 (いいね順)")
                st.dataframe(
                    display_en[["comment", "likes", "gemini_category", "gemini_sentiment"]],
                    column_config={
                        "comment": st.column_config.TextColumn("コメント内容", width="large"),
                        "likes": st.column_config.NumberColumn("いいね", width="small"),
                        "gemini_category": st.column_config.TextColumn("感情カテゴリ", width="medium"),
                        "gemini_sentiment": st.column_config.NumberColumn("スコア", format="%.2f"),
                    },
                    use_container_width=True
                )
            else:
                st.info("英語コメントはありません。")

        # 🇯🇵 日本語
        with col2:
            st.header("日本語コメント分析 (Gemini推論)")
            if not df_ja.empty:
                avg_ja = df_ja["gemini_sentiment"].mean()
                weighted_ja = calculate_weighted_sentiment(df_ja["gemini_sentiment"], df_ja["likes"])
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("単純平均スコア", f"{avg_ja:.2f}")
                with m2:
                    st.metric("世論支持スコア (加重平均)", f"{weighted_ja:.2f}", delta=f"{weighted_ja - avg_ja:+.2f}")
                
                fig_pie_ja = px.pie(
                    df_ja, names="gemini_category", title="感情の割合 (日本語: Gemini)",
                    color="gemini_category", color_discrete_map=color_map, hole=0.3
                )
                st.plotly_chart(fig_pie_ja, use_container_width=True)
                
                fig_hist_ja = px.histogram(
                    df_ja, x="gemini_sentiment", nbins=20,
                    title="📊 感情スコア分布 (日本語: -1.0〜+1.0)",
                    labels={"gemini_sentiment": "感情スコア"},
                    color_discrete_sequence=["#d7191c"]
                )
                st.plotly_chart(fig_hist_ja, use_container_width=True)
                
                st.subheader("🔤 よく使われている日本語単語 Top 10")
                top_words_ja = get_top_words_ja(df_ja["comment"])
                if not top_words_ja.empty:
                    fig_bar_ja = px.bar(top_words_ja, x="出現回数", y="単語", orientation="h", text="出現回数")
                    fig_bar_ja.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_bar_ja, use_container_width=True)
                
                st.subheader("💬 日本語コメント一覧・絞り込み")
                filter_ja = st.radio(
                    "表示する感情を選択 (日本語):",
                    ["すべて表示", "Positive (肯定的)", "Neutral (中立)", "Negative (否定的)"],
                    key="filter_ja_tab1",
                    horizontal=True
                )
                if filter_ja == "すべて表示":
                    display_ja = df_ja.sort_values(by="likes", ascending=False)
                else:
                    display_ja = df_ja[df_ja["gemini_category"] == filter_ja].sort_values(by="likes", ascending=False)
                    
                st.caption(f"表示中: {len(display_ja)} 件 (いいね順)")
                st.dataframe(
                    display_ja[["comment", "likes", "gemini_category", "gemini_sentiment"]],
                    column_config={
                        "comment": st.column_config.TextColumn("コメント内容", width="large"),
                        "likes": st.column_config.NumberColumn("いいね", width="small"),
                        "gemini_category": st.column_config.TextColumn("感情カテゴリ", width="medium"),
                        "gemini_sentiment": st.column_config.NumberColumn("スコア", format="%.2f"),
                    },
                    use_container_width=True
                )
            else:
                st.info("日本語コメントはありません。")

    # ==========================================
    # TAB 2: BERT vs Gemini 判定差分ビューア
    # ==========================================
    with tab2:
        st.header("🔬 モデル比較: ローカルBERT vs Gemini 3.6 Flash")
        if not df_ja.empty:
            match_count = df_ja["is_match"].sum()
            total_ja = len(df_ja)
            match_rate = (match_count / total_ja) * 100 if total_ja > 0 else 0
            
            rescued_df = df_ja[
                (df_ja["bert_category"] == "Negative (否定的)") & 
                (df_ja["gemini_category"] == "Positive (肯定的)")
            ]
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("両モデルの一致率", f"{match_rate:.1f}%", f"{match_count}/{total_ja} 件")
            with sc2:
                st.metric("共感救済コメント数 (BERT否定 ➔ Gemini肯定)", f"{len(rescued_df)} 件", help="単語に引きずられた否定判定が、LLMで正しく肯定に救済されたコメントです。")
            with sc3:
                pos_diff = (df_ja["gemini_category"] == "Positive (肯定的)").mean() - (df_ja["bert_category"] == "Positive (肯定的)").mean()
                st.metric("肯定判定割合の差分 (Gemini - BERT)", f"{pos_diff*100:+.1f}%")
            
            st.markdown("---")
            
            st.subheader("📊 判定割合の直接比較")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                fig_pie_bert = px.pie(
                    df_ja, names="bert_category", title="🤖 ローカルBERTの感情割合",
                    color="bert_category", color_discrete_map=color_map, hole=0.3
                )
                st.plotly_chart(fig_pie_bert, use_container_width=True)

            with g_col2:
                fig_pie_gemini = px.pie(
                    df_ja, names="gemini_category", title="✨ Gemini 3.6 Flash（文脈考慮）の感情割合",
                    color="gemini_category", color_discrete_map=color_map, hole=0.3
                )
                st.plotly_chart(fig_pie_gemini, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 日本語コメント判定の差分絞り込み")
            
            diff_filter = st.radio(
                "表示するコメントの種別を選択:",
                [
                    "💡 共感救済例（BERT否定 ➔ Gemini肯定）",
                    "⚠️ 判定不一致のコメントすべて",
                    "✅ 判定一致のコメント",
                    "すべて表示"
                ],
                horizontal=True
            )
            
            if diff_filter == "💡 共感救済例（BERT否定 ➔ Gemini肯定）":
                display_diff_df = rescued_df
            elif diff_filter == "⚠️ 判定不一致のコメントすべて":
                display_diff_df = df_ja[~df_ja["is_match"]]
            elif diff_filter == "✅ 判定一致のコメント":
                display_diff_df = df_ja[df_ja["is_match"]]
            else:
                display_diff_df = df_ja
                
            st.caption(f"表示中: {len(display_diff_df)} 件 (いいね順)")
            
            st.dataframe(
                display_diff_df[[
                    "comment", "likes", "bert_category", "gemini_category", 
                    "bert_sentiment", "gemini_sentiment"
                ]].sort_values(by="likes", ascending=False),
                column_config={
                    "comment": st.column_config.TextColumn("コメント内容", width="large"),
                    "likes": st.column_config.NumberColumn("いいね", width="small"),
                    "bert_category": st.column_config.TextColumn("BERT判定", width="medium"),
                    "gemini_category": st.column_config.TextColumn("Gemini判定", width="medium"),
                    "bert_sentiment": st.column_config.NumberColumn("BERTスコア", format="%.2f"),
                    "gemini_sentiment": st.column_config.NumberColumn("Geminiスコア", format="%.2f"),
                },
                use_container_width=True
            )
        else:
            st.info("日本語コメントのデータがありません。")