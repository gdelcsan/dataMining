import streamlit as st
import pandas as pd
import numpy as np
import time
from itertools import combinations, chain
from pathlib import Path

# ------------------------------
# Helpers
# ------------------------------

def load_default_csv(name: str) -> pd.DataFrame | None:
    """Tries to load a CSV bundled in this environment (classroom uploads)."""
    p = Path('/mnt/data')/name
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def normalize_item(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return " ".join(x.strip().lower().split())


def preprocess_transactions(df: pd.DataFrame, products_df: pd.DataFrame):
    """Perform required cleaning and return cleaned transactions + report dict."""
    # Expect columns: Transaction / Items Purchased  OR  a single column 'items' with comma sep
    if df.shape[1] == 1:
        df = df.copy()
        df.columns = ['items']
    else:
        # heuristics: use last non-empty column as items column
        items_col = df.columns[-1]
        df = df[[items_col]].rename(columns={items_col: 'items'})

    before_total = len(df)

    # split comma separated into lists
    tx_lists = []
    for raw in df['items'].astype(str).fillna(""):
        if "," in raw:
            tx_lists.append([normalize_item(x) for x in raw.split(',') if normalize_item(x)])
        else:
            # space-separated or single token
            parts = [normalize_item(x) for x in raw.split(' ') if normalize_item(x)]
            tx_lists.append(parts)

    # remove empties
    empty_count = sum(1 for t in tx_lists if len(t) == 0)
    tx_lists = [t for t in tx_lists if len(t) > 0]

    # standardize case/whitespace already done; remove duplicates within each transaction
    dup_instances = 0
    deduped = []
    for t in tx_lists:
        seen = []
        for it in t:
            if it not in seen:
                seen.append(it)
            else:
                dup_instances += 1
        deduped.append(seen)

    # single-item handling: remove
    single_count = sum(1 for t in deduped if len(t) == 1)
    deduped = [t for t in deduped if len(t) > 1]

    # invalid product handling using products_df (if provided)
    invalid_instances = 0
    valid_names = None
    if products_df is not None and not products_df.empty:
        cols = [c.lower() for c in products_df.columns]
        products_df.columns = cols
        name_col = 'name' if 'name' in cols else (cols[-1] if cols else None)
        if name_col is not None:
            valid_names = set(normalize_item(x) for x in products_df[name_col].astype(str))
    cleaned = []
    for t in deduped:
        if valid_names is None:
            cleaned.append(t)
            continue
        keep = [it for it in t if it in valid_names]
        invalid_instances += len(t) - len(keep)
        if len(keep) > 1:
            cleaned.append(keep)
    
    after_total = len(cleaned)
    total_items = sum(len(t) for t in cleaned)
    unique_products = len(set(chain.from_iterable(cleaned)))

    report = {
        'before_total_tx': before_total,
        'empty_tx_removed': empty_count,
        'single_item_tx_removed': single_count,
        'duplicate_items_removed': dup_instances,
        'invalid_items_removed': invalid_instances,
        'after_valid_tx': after_total,
        'total_items': total_items,
        'unique_products': unique_products,
    }
    return cleaned, report


# ------------------------------
# Apriori (from scratch)
# ------------------------------

def get_support(itemset, tx_list):
    count = 0
    s = set(itemset)
    for t in tx_list:
        if s.issubset(t):
            count += 1
    return count / len(tx_list) if tx_list else 0.0


def apriori(transactions, min_support=0.2):
    """Return dict: {k: {frozenset(items): support}} for each size k>=1."""
    # L1
    item_counts = {}
    n_tx = len(transactions)
    for t in transactions:
        for it in set(t):
            item_counts[it] = item_counts.get(it, 0) + 1
    L = {}
    L1 = {frozenset([it]): c/n_tx for it, c in item_counts.items() if c/n_tx >= min_support}
    if not L1:
        return {}
    L[1] = L1

    k = 2
    current_L = L1
    while current_L:
        # generate candidates by self-join
        cand = set()
        keys = list(current_L.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                union = keys[i] | keys[j]
                if len(union) == k:
                    # prune: all (k-1)-subsets must be frequent
                    all_subfreq = all((union - frozenset([x])) in current_L for x in union)
                    if all_subfreq:
                        cand.add(union)
        # count support
        Ck = {}
        for c in cand:
            sup = get_support(c, transactions)
            if sup >= min_support:
                Ck[c] = sup
        if Ck:
            L[k] = Ck
            current_L = Ck
            k += 1
        else:
            break
    return L


def generate_rules(freq_dict, min_conf=0.5, n_tx=1):
    """Generate association rules (A -> B) with confidence >= min_conf.
    Returns list of dicts: {antecedent, consequent, support, confidence, lift}.
    """
    # build quick support lookup
    sup_lookup = {}
    for k, m in freq_dict.items():
        for iset, sup in m.items():
            sup_lookup[iset] = sup

    rules = []
    for k, m in freq_dict.items():
        if k < 2:
            continue
        for iset, sup_ab in m.items():
            items = list(iset)
            # all non-empty proper subsets as antecedents
            for r in range(1, len(items)):
                for A in combinations(items, r):
                    A = frozenset(A)
                    B = iset - A
                    sup_a = sup_lookup.get(A, 0)
                    sup_b = sup_lookup.get(B, 0)
                    if sup_a == 0 or len(B) == 0:
                        continue
                    conf = sup_ab / sup_a
                    if conf >= min_conf:
                        lift = conf / sup_b if sup_b > 0 else np.nan
                        rules.append({
                            'antecedent': tuple(sorted(A)),
                            'consequent': tuple(sorted(B)),
                            'support': sup_ab,
                            'confidence': conf,
                            'lift': lift
                        })
    # sort by confidence desc, then lift desc
    rules.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)
    return rules


# ------------------------------
# Eclat (from scratch)
# ------------------------------

def build_vertical_format(transactions):
    """Return dict item -> TID set."""
    vert = {}
    for tid, t in enumerate(transactions):
        for it in set(t):
            vert.setdefault(frozenset([it]), set()).add(tid)
    return vert


def eclat_recursive(prefix, items_tidsets, min_support, n_tx, out):
    while items_tidsets:
        (item, tidset) = items_tidsets.pop()
        new_prefix = prefix | item
        support = len(tidset)/n_tx if n_tx else 0
        if support >= min_support:
            out[new_prefix] = support
            # intersect with remaining to build extensions
            new_items = []
            for (item2, tidset2) in items_tidsets:
                inter = tidset & tidset2
                if inter:
                    new_items.append((item2, inter))
            eclat_recursive(new_prefix, new_items, min_support, n_tx, out)


def eclat(transactions, min_support=0.2):
    vert = build_vertical_format(transactions)
    items = list(vert.items())
    out = {}
    eclat_recursive(frozenset(), items, min_support, len(transactions), out)
    # group by k
    by_k = {}
    for iset, sup in out.items():
        by_k.setdefault(len(iset), {})[iset] = sup
    return by_k


# ------------------------------
# UI
# ------------------------------

st.set_page_config(page_title="Supermarket Miner", page_icon="🛒", layout="wide")
st.title("🛒 Interactive Supermarket Simulation + Association Rule Mining")

with st.sidebar:
    st.header("Data Sources")
    st.caption("Upload your s or use the provided classroom defaults if available.")
    up_tx = st.file_uploader("Transactions (comma-separated items per row)", type=None, key="tx_upload")
    up_prod = st.file_uploader("Products (valid product names)", type=None, key="prod_upload")

    default_tx = load_default_('./assignment_data_mining/sample_transactions.csv')
    default_prod = load_default_csv('./assignment_data_mining/products.csv')

    if up_tx is not None:
        tx_df_raw = pd.read_csv(up_tx)
    elif default_tx is not None:
        tx_df_raw = default_tx
    else:
        tx_df_raw = pd.DataFrame({'items': []})

    if up_prod is not None:
        prod_df_raw = pd.read_csv(up_prod)
    else:
        prod_df_raw = default_prod if default_prod is not None else pd.DataFrame({'name': []})

    st.divider()
    st.header("Mining Parameters")
    min_support = st.slider("Minimum Support", 0.05, 0.9, 0.2, 0.05)
    min_conf = st.slider("Minimum Confidence", 0.05, 0.95, 0.5, 0.05)

# Session state for manual transactions
if 'manual_txs' not in st.session_state:
    st.session_state.manual_txs = []

# Product palette
if prod_df_raw is not None and not prod_df_raw.empty:
    cols = [c.lower() for c in prod_df_raw.columns]
    prod_df_raw.columns = cols
    name_col = 'name' if 'name' in cols else cols[-1]
    product_names = sorted({normalize_item(x) for x in prod_df_raw[name_col].astype(str) if normalize_item(x)})
else:
    product_names = [
        'milk','bread','eggs','butter','cheese','apples','bananas','cereal','coffee','tea',
        'yogurt','juice','chicken','beef','rice','pasta','tomato','onion','lettuce','cookies'
    ]

st.subheader("1) Create Transactions Manually")
col1, col2 = st.columns([2,1])
with col1:
    sel = st.multiselect("Select products to add as a transaction:", options=product_names, key="picker")
    add = st.button("➕ Add Transaction", type="primary")
    if add and sel:
        norm = [normalize_item(x) for x in sel if normalize_item(x)]
        norm = sorted(set(norm))
        if len(norm) > 1:
            st.session_state.manual_txs.append(norm)
        else:
            st.warning("Single-item transactions are ignored for mining.")
with col2:
    if st.button("🧹 Clear Manual Transactions"):
        st.session_state.manual_txs = []

# Show raw imported transactions sample
st.subheader("2) Imported Transactions (raw preview)")
st.dataframe(tx_df_raw.head(12), use_container_width=True)

# Combine imported + manual for preprocessing
combined_df = tx_df_raw.copy()
if st.session_state.manual_txs:
    extra = pd.DataFrame({'items': [", ".join(t) for t in st.session_state.manual_txs]})
    if 'items' in combined_df.columns:
        combined_df = pd.concat([combined_df[['items']], extra], ignore_index=True)
    else:
        # place manual only
        combined_df = extra

run_prep = st.button("🚿 Run Preprocessing")
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = None
    st.session_state.report = None

if run_prep:
    cleaned, report = preprocess_transactions(combined_df, prod_df_raw)
    st.session_state.cleaned = [set(t) for t in cleaned]
    st.session_state.report = report

if st.session_state.cleaned is not None:
    st.success("Preprocessing complete.")
    with st.expander("Preprocessing Report", expanded=True):
        r = st.session_state.report
        left, right = st.columns(2)
        with left:
            st.metric("Total transactions (before)", r['before_total_tx'])
            st.metric("Empty transactions removed", r['empty_tx_removed'])
            st.metric("Single-item tx removed", r['single_item_tx_removed'])
        with right:
            st.metric("Duplicate items removed", r['duplicate_items_removed'])
            st.metric("Invalid items removed", r['invalid_items_removed'])
            st.metric("Valid transactions (after)", r['after_valid_tx'])
        st.caption(f"Total items: {r['total_items']} • Unique products: {r['unique_products']}")

    st.subheader("Cleaned Transactions (sample)")
    sample = [' , '.join(sorted(t)) for t in st.session_state.cleaned[:25]]
    st.dataframe(pd.DataFrame({'transaction': sample}), use_container_width=True, hide_index=True)

st.divider()

# Mining section
st.subheader("3) Run Mining (Apriori & Eclat)")
run_mining = st.button("🧠 Analyze")

if 'results' not in st.session_state:
    st.session_state.results = {}

if run_mining:
    if not st.session_state.cleaned:
        st.error("Please run preprocessing first (and ensure you have at least 2-item transactions).")
    else:
        tx = [set(t) for t in st.session_state.cleaned]
        # Apriori
        t0 = time.perf_counter()
        L_ap = apriori(tx, min_support=min_support)
        rules_ap = generate_rules(L_ap, min_conf=min_conf, n_tx=len(tx))
        t1 = time.perf_counter()
        # Eclat
        t2 = time.perf_counter()
        L_ec = eclat(tx, min_support=min_support)
        rules_ec = generate_rules(L_ec, min_conf=min_conf, n_tx=len(tx))
        t3 = time.perf_counter()

        st.session_state.results = {
            'apriori': {
                'freq': L_ap,
                'rules': rules_ap,
                'runtime_ms': (t1 - t0)*1000
            },
            'eclat': {
                'freq': L_ec,
                'rules': rules_ec,
                'runtime_ms': (t3 - t2)*1000
            },
            'n_tx': len(tx)
        }

if st.session_state.get('results'):
    res = st.session_state.results
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Apriori**")
        st.write(f"Runtime: {res['apriori']['runtime_ms']:.1f} ms")
        st.write(f"Rules generated: {len(res['apriori']['rules'])}")
    with c2:
        st.markdown("**Eclat**")
        st.write(f"Runtime: {res['eclat']['runtime_ms']:.1f} ms")
        st.write(f"Rules generated: {len(res['eclat']['rules'])}")

    # Display rules (toggle technical)
    with st.expander("Show technical rules (Apriori)"):
        st.dataframe(pd.DataFrame(res['apriori']['rules']), use_container_width=True)
    with st.expander("Show technical rules (Eclat)"):
        st.dataframe(pd.DataFrame(res['eclat']['rules']), use_container_width=True)

    st.subheader("4) Query Recommendations")
    all_items = sorted(set(chain.from_iterable([list(s) for s in (res['apriori']['freq'].get(1, {}) or {}).keys()]))
                     if res['apriori']['freq'] else sorted(product_names))
    picked = st.selectbox("Pick a product to see associated items:", options=all_items)

    def recommendations_for(item, rules):
        agg = {}
        for r in rules:
            if item in r['antecedent']:
                for c in r['consequent']:
                    best = agg.get(c)
                    score = r['confidence']
                    if best is None or score > best['confidence']:
                        agg[c] = {
                            'confidence': score,
                            'support': r['support'],
                            'lift': r['lift']
                        }
        # turn into list
        out = [
            {'item': k, 'confidence_pct': v['confidence']*100, 'support_pct': v['support']*100, 'lift': v['lift']}
            for k, v in agg.items()
        ]
        out.sort(key=lambda x: (x['confidence_pct'], x['lift']), reverse=True)
        return out

    if picked:
        ap_recs = recommendations_for(picked, res['apriori']['rules'])
        ec_recs = recommendations_for(picked, res['eclat']['rules'])
        tab1, tab2 = st.tabs(["Apriori", "Eclat"])
        with tab1:
            if ap_recs:
                df = pd.DataFrame(ap_recs)
                df['strength'] = pd.cut(df['confidence_pct'], bins=[0,40,70,100], labels=["Weak","Moderate","Strong"], include_lowest=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.write(f"**Recommendation:** Consider bundling **{picked}** with the top 1–2 items above.")
            else:
                st.info("No associations found for this item at current thresholds.")
        with tab2:
            if ec_recs:
                df = pd.DataFrame(ec_recs)
                df['strength'] = pd.cut(df['confidence_pct'], bins=[0,40,70,100], labels=["Weak","Moderate","Strong"], include_lowest=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.write(f"**Recommendation:** Consider placement and promotions pairing **{picked}** with the top items.")
            else:
                st.info("No associations found for this item at current thresholds.")

st.divider()

st.caption("Tip: Save this file as `streamlit_app.py` and run locally with `streamlit run streamlit_app.py`.")
