import numpy as np
import pandas as pd

def build_event_features_advanced(df, events, extra_v2=False):
    base = df[["lead_id", "assignment_ts"]].copy()
    base["assignment_ts"] = pd.to_datetime(base["assignment_ts"])

    ev = events.copy()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    ev["is_weekend"] = ev["event_ts"].dt.weekday.isin([5, 6]).astype(int)
    ev["hour"] = ev["event_ts"].dt.hour

    merged = ev.merge(base, on="lead_id")
    merged = merged[merged["event_ts"] < merged["assignment_ts"]].copy()
    merged["hours_before"] = (merged["assignment_ts"] - merged["event_ts"]).dt.total_seconds() / 3600

    frames = []
    windows = {"12h": 12, "1d": 24, "3d": 72, "7d": 168, "14d": 336, "30d": 720}

    # 1. Счётчики по окнам
    for name, h in windows.items():
        w = merged[merged["hours_before"] <= h]
        cnt = w.groupby(["lead_id", "event_type"]).size().unstack(fill_value=0)
        cnt.columns = [f"ev_{c}_c_{name}" for c in cnt.columns]
        frames.append(cnt)

    # 2. Recency
    rec = merged.groupby(["lead_id", "event_type"])["hours_before"].min().unstack()
    rec.columns = [f"ev_{c}_r" for c in rec.columns]
    frames.append(rec)

    # 3. Общие агрегаты
    agg = merged.groupby("lead_id").agg(
        ev_cnt=("event_type", "count"),
        ev_ctx=("ctx_seq", "nunique"),
        ev_slot=("src_slot", "nunique"),
        ev_unique_days=("event_ts", lambda x: x.dt.date.nunique()),
        ev_unique_types=("event_type", "nunique"),
        ev_weekend_ratio=("is_weekend", "mean"),
        ev_avg_interval=("hours_before", lambda x: x.sort_values().diff().mean() if len(x)>1 else 9999),
        ev_first_hours=("hours_before", "max"),
        ev_last_hours=("hours_before", "min"),
        ev_price_mean=("item_price_log", "mean"),
        ev_price_std=("item_price_log", "std"),
        ev_price_max=("item_price_log", "max"),
        ev_price_min=("item_price_log", "min"),
    )
    frames.append(agg)

    # 4. Агрегаты по src_slot и ctx_seq
    slot_agg = merged.groupby("lead_id").agg(
        slot_nunique=("src_slot", "nunique"),
        slot_mode=("src_slot", lambda x: x.mode()[0] if not x.mode().empty else -1),
        slot_mode_ratio=("src_slot", lambda x: (x.value_counts().iloc[0] / len(x)) if len(x)>0 else 0),
        ctx_nunique=("ctx_seq", "nunique"),
        ctx_mode=("ctx_seq", lambda x: x.mode()[0] if not x.mode().empty else "missing"),
        ctx_mode_ratio=("ctx_seq", lambda x: (x.value_counts().iloc[0] / len(x)) if len(x)>0 else 0),
    )
    frames.append(slot_agg)

    # 5. Бинарные флаги наличия событий в окнах
    for name, h in windows.items():
        has = merged[merged["hours_before"] <= h].groupby("lead_id").size().gt(0).astype(int)
        has.name = f"ev_has_{name}"
        frames.append(has.to_frame())

    # 6. Соотношения основных типов событий
    event_counts = merged.groupby(["lead_id", "event_type"]).size().unstack(fill_value=0)
    total_events = event_counts.sum(axis=1)
    for col in ["item_view", "search", "favorite", "chat_open", "call_click", "detail_expand", "photo_swipe"]:
        if col in event_counts.columns:
            event_counts[f"ratio_{col}_to_total"] = event_counts[col] / (total_events + 1e-6)
    frames.append(event_counts[[c for c in event_counts.columns if "ratio" in c]])

    # ctx_seq share distribution 
    ctx_pivot = merged.groupby(["lead_id", "ctx_seq"]).size().unstack(fill_value=0)
    ctx_pivot = ctx_pivot.div(ctx_pivot.sum(axis=1).replace(0, np.nan), axis=0)
    ctx_pivot.columns = [f"ctx_share_{c}" for c in ctx_pivot.columns]
    frames.append(ctx_pivot)

    # decay-взвешенная активность 
    for half_life in [24, 168]:
        merged[f"_decay_{half_life}"] = np.exp(-np.log(2) * merged["hours_before"] / half_life)
        decay_sum = merged.groupby("lead_id")[f"_decay_{half_life}"].sum()
        decay_sum.name = f"ev_decay_sum_{half_life}h"
        frames.append(decay_sum.to_frame())

    if extra_v2:
        # session count: gap > 6h between consecutive events = new session
        def count_sessions(x):
            if len(x) == 0:
                return 0
            s = x.sort_values().values
            gaps = np.diff(s)
            return 1 + (gaps > 6).sum()
        
        sessions = merged.groupby("lead_id")["hours_before"].apply(lambda x: count_sessions(-x)).rename("n_sessions")
        frames.append(sessions.to_frame())

        # decay momentum ratio (short vs long decay)
        decay24 = merged.groupby("lead_id")["_decay_24"].sum()
        decay168 = merged.groupby("lead_id")["_decay_168"].sum()
        decay_ratio = (decay24 / (decay168 + 1e-6)).rename("decay_momentum_ratio")
        frames.append(decay_ratio.to_frame())

    result = pd.concat(frames, axis=1).reindex(base["lead_id"]).reset_index()
    result = result.rename(columns={"index": "lead_id"})
    result = result.fillna(0)

    # Recency missing -> 9999
    rec_cols = [c for c in result.columns if "_r" in c]
    result[rec_cols] = result[rec_cols].fillna(9999)

    # Последние ctx_seq и src_slot
    last_event = merged.sort_values("hours_before").groupby("lead_id").first()
    last_cols = ["ctx_seq", "src_slot"]
    rename_map = {"ctx_seq": "last_ctx_seq", "src_slot": "last_src_slot"}
    
    if extra_v2:
        last_cols.append("event_type")
        rename_map["event_type"] = "last_event_type"
        
    last_ctx = last_event[last_cols].rename(columns=rename_map)
    result = result.merge(last_ctx, on="lead_id", how="left")
    
    result["last_ctx_seq"] = result["last_ctx_seq"].fillna("missing")
    result["last_src_slot"] = result["last_src_slot"].fillna(-1)
    if extra_v2:
        result["last_event_type"] = result["last_event_type"].fillna("missing")

    return result

def add_business_features(df):
    df = df.copy()
    df["user_active_ratio"] = df["user_active_days_30d"] / (df["user_age_days"] + 1)
    df["user_activity_per_lead"] = df["user_active_days_30d"] / (df["prior_assignments_30d"] + 1)
    df["prior_success_ratio_30d"] = df["leadgen_prev_positive_30d"] / (df["leadgen_prev_assigned_30d"] + 1)
    df["prior_answer_ratio_30d"] = df["leadgen_prev_answered_30d"] / (df["leadgen_prev_assigned_30d"] + 1)
    df["prior_experience"] = df["prior_assignments_30d"] / (df["user_age_days"] + 1)
    df["seller_quality"] = df["seller_response_rate_30d"] * df["seller_inventory_count"]
    df["seller_activity"] = df["seller_page_views_30d"] / (df["seller_inventory_count"] + 1)
    df["price_per_year"] = df["item_price_log"] / (df["car_age_years"] + 1)
    df["mileage_per_year"] = df["mileage_km_log"] / (df["car_age_years"] + 1)
    df["price_per_mileage"] = df["item_price_log"] / (df["mileage_km_log"] + 1)
    df["car_age_log"] = np.log1p(df["car_age_years"])
    df["views_per_day_30d"] = df["item_views_30d"] / 30
    df["views_per_day_7d"] = df["item_views_7d"] / 7
    df["favorites_per_view_30d"] = df["item_favorites_30d"] / (df["item_views_30d"] + 1)
    df["detail_expands_per_view_30d"] = df["detail_expands_30d"] / (df["item_views_30d"] + 1)
    df["photo_swipes_per_view_30d"] = df["photo_swipes_30d"] / (df["item_views_30d"] + 1)
    df["search_per_day_30d"] = df["search_views_30d"] / 30
    df["search_to_views_ratio"] = df["search_views_30d"] / (df["item_views_30d"] + 1)
    df["refinements_per_search"] = df["query_refinements_30d"] / (df["search_views_30d"] + 1)
    df["contacts_per_view_30d"] = df["user_contacts_30d"] / (df["item_views_30d"] + 1)
    
    df["chat_per_contact_30d"] = df["chat_opens_30d"] / (df["user_contacts_30d"] + 1)
    df["call_per_contact_30d"] = df["call_clicks_30d"] / (df["user_contacts_30d"] + 1)
    df["chat_call_ratio_30d"] = df["chat_opens_30d"] / (df["call_clicks_30d"] + 1)

    df["views_trend_7d_to_30d"] = df["item_views_7d"] / (df["item_views_30d"] + 1)
    df["views_trend_1d_to_7d"] = df["item_views_1d"] / (df["item_views_7d"] + 1)

    df["contacts_trend_7d_to_30d"] = df["user_contacts_7d"] / (df["user_contacts_30d"] + 1)
    df["search_trend_7d_to_30d"] = df["search_views_7d"] / (df["search_views_30d"] + 1)
    df["age_price_interaction"] = df["user_age_days"] * df["item_price_log"]
    df["inventory_price_interaction"] = df["seller_inventory_count"] * df["item_price_log"]
    df["activity_seller_interaction"] = df["user_active_days_30d"] * df["seller_response_rate_30d"]

    df["user_total_views_30d"] = df["item_views_30d"] + df["search_views_30d"] + df["seller_page_views_30d"]
    df["user_total_actions_30d"] = df["user_total_views_30d"] + df["user_contacts_30d"] + df["chat_opens_30d"]
    df["user_engagement_score"] = (df["item_favorites_30d"] + df["detail_expands_30d"] + df["photo_swipes_30d"]) / (df["item_views_30d"] + 1)

    df["price_pct_in_segment"] = df.groupby("car_segment")["item_price_log"].rank(pct=True)
    df["price_pct_in_region"] = df.groupby("region")["item_price_log"].rank(pct=True)

    return df