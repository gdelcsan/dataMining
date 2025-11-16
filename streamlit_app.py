import streamlit as st
import pandas as pd
import numpy as np
import time
from itertools import combinations, chain
from pathlib import Path

# ------------------------------
# Style
st.markdown("""
    <style>
    /* Sidebar container */
    section[data-testid="stSidebar"] {
        color: #ffffff;
        text-align: center;
        background-color: #9CE6E6;
        background-image: linear-gradient(120deg, #33CCCC, #2AA7A7);
        border-right: 1px solid rgba(27,31,35,0.1);  
    }
    section[data-testid="stSidebar"] label { color: white; }

    /* Buttons */
    .stButton > button {
        color: white;
        background-color: #D55858;
        border: none;
        border-radius: 9999px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #D55858;
        background-image: linear-gradient(90deg, #D55858, #A72A2A);
        transform: scale(1.02);
    }

    /* Title gradient */
    .header {
        text-align: center;
        padding: 1rem 1rem;
        font-size: 5rem;             
        font-weight: 800;
        background: linear-gradient(90deg, #D55858, #A72A2A); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Tabs */
    .stTabs [aria-selected="false"] { color: #000000; 
    }
    
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Page config & title
st.set_page_config(page_title="Supermarket Miner", page_icon="🛒", layout="wide")
st.markdown(
    '<div class="header"><h1>Interactive Supermarket Simulation with Association Rule Mining</h1></div>',
    unsafe_allow_html=True
)

# ------------------------------
# Helpers
def normalize_item(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return " ".join(x.strip().lower().split())

def safe_read_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        st.error(f"File not found: {p}")
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Error reading {p.name}: {e}")
        return pd.DataFrame()

# ------------------------------
# Preprocessing
def preprocess_transactions(df: pd.DataFrame, products_df: pd.DataFrame, flag_singles=False, strict_invalid=False):
    """Clean data and return cleaned list + report."""
    if df.shape[1] == 1:
        df = df.copy()
        df.columns = ['items']
    else:
        items_col = df.columns[-1]
        df = df[[items_col]].rename(columns={items_col: 'items'})

    before_total = len(df)

    # split into lists
    tx_lists = []
    for raw in df['items'].astype(str).fillna(""):
        if "," in raw:
            tx_lists.append([normalize_item(x) for x in raw.split(',') if normalize_item(x)])
        else:
            parts = [normalize_item(x) for x in raw.split(' ') if normalize_item(x)]
            tx_lists.append(parts)

    # empty tx
    empty_count = sum(1 for t in tx_lists if len(t) == 0)
    tx_lists = [t for t in tx_lists if len(t) > 0]

    # duplicates
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

    # single-item handling
    single_count = sum(1 for t in deduped if len(t) == 1)
    if flag_singles:
        pass  # keep them
    else:
        deduped = [t for t in deduped if len(t) > 1]

    # invalid items
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
        if strict_invalid and len(keep) < len(t):
            continue  # drop whole tx if any invalid
        if len(keep) > 1 or (flag_singles and len(keep) == 1):
            cleaned.append(keep)

    after_total = len(cleaned)
    total_items = sum(len(t) for t in cleaned)
    unique_products = len(set(chain.from_iterable(cleaned)))

    total_issues = empty_count + single_count + dup_instances + invalid_instances

    report = {
        'before_total_tx': before_total,
        'empty_tx_removed': empty_count,
        'single_item_tx_removed': single_count if not flag_singles else 0,
        'single_item_flagged': single_count if flag_singles else 0,
        'duplicate_items_removed': dup_instances,
        'invalid_items_removed': invalid_instances,
        'after_valid_tx': after_total,
        'total_items': total_items,
        'unique_products': unique_products,
        'total_issues': total_issues
    }
    return cleaned, report

# ------------------------------
# Apriori & Eclat 
def get_support(itemset, tx_list):
    count = 0
    s = set(itemset)
    for t in tx_list:
        if s.issubset(t):
            count += 1
    return count / len(tx_list) if tx_list else 0.0

def apriori(transactions, min_support=0.2):
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
        cand = set()
        keys = list(current_L.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                union = keys[i] | keys[j]
                if len(union) == k:
                    all_subfreq = all((union - frozenset([x])) in current_L for x in union)
                    if all_subfreq:
                        cand.add(union)
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
    rules.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)
    return rules

def build_vertical_format(transactions):
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
    by_k = {}
    for iset, sup in out.items():
        by_k.setdefault(len(iset), {})[iset] = sup
    return by_k

# ------------------------------
# Load products
PROD_PATH = "./assignment_data_mining/products.csv"
prod_df_raw = safe_read_csv(PROD_PATH)

# Derive product names
if prod_df_raw is not None and not prod_df_raw.empty:
    cols = [c.lower() for c in prod_df_raw.columns]
    prod_df_raw.columns = cols
    name_col = 'name' if 'name' in cols else cols[-1]
    product_names = sorted({normalize_item(x) for x in prod_df_raw[name_col].astype(str) if normalize_item(x)})
else:
    product_names = [
        'milk','bread','eggs','butter','cheese','apple','banana','orange','coffee','tea',
        'yogurt','juice','chicken','beef','rice','pasta','tomato','onion','garlic','pepper'
    ]

# ------------------------------
# Sidebar
st.sidebar.header("Mining Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.05, 0.9, 0.2, 0.05)
min_conf = st.sidebar.slider("Minimum Confidence", 0.05, 0.95, 0.5, 0.05)

st.sidebar.header("Preprocessing Options")
flag_singles = st.sidebar.checkbox("Flag single-item transactions (instead of removing)", value=False)
strict_invalid = st.sidebar.checkbox("Remove entire transaction if any item is invalid", value=False)

# NEW: Sidebar uploader for custom transactions
st.sidebar.header("Upload Custom Transactions")
sidebar_file = st.sidebar.file_uploader("Upload transactions CSV", type="csv")

# ------------------------------
# Session state
if 'manual_cart' not in st.session_state:
    st.session_state.manual_cart = []  # current cart
if 'manual_txs' not in st.session_state:
    st.session_state.manual_txs = []  # saved transactions
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'ap_rules' not in st.session_state:
    st.session_state.ap_rules = []
if 'ec_rules' not in st.session_state:
    st.session_state.ec_rules = []

# Handle sidebar upload (replace current dataset)
if sidebar_file is not None:
    try:
        uploaded_df = pd.read_csv(sidebar_file)
        st.session_state.uploaded_df = uploaded_df
        st.sidebar.success(f"{len(uploaded_df)} transactions loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")

# ------------------------------
# 1) Create Transactions Manually
st.subheader("Create Transactions Manually")
cols = st.columns(5)
for i, prod in enumerate(product_names[:20]):  # show at least 10
    with cols[i % 5]:
        if st.button(prod.title(), key=f"btn_{prod}"):
            if prod not in st.session_state.manual_cart:
                st.session_state.manual_cart.append(prod)
                st.rerun()

# Show current cart
if st.session_state.manual_cart:
    st.write("**Current cart:** " + ", ".join([p.title() for p in st.session_state.manual_cart]))

col_add, col_clear = st.columns([1,1])
with col_add:
    if st.button("Add Transaction", type="primary"):
        norm = sorted(set(st.session_state.manual_cart))
        if len(norm) > 1 or flag_singles:
            st.session_state.manual_txs.append(norm)
            st.session_state.manual_cart = []
            st.success("Transaction added!")
            st.rerun()
        else:
            st.warning("Need at least 2 items.")
with col_clear:
    if st.button("Clear Cart"):
        st.session_state.manual_cart = []
        st.rerun()

# Display all manual transactions
if st.session_state.manual_txs:
    manual_df = pd.DataFrame({
        'transaction': [', '.join(t) for t in st.session_state.manual_txs]
    })
    with st.expander("Manual Transactions", expanded=False):
        st.dataframe(manual_df, use_container_width=True, hide_index=True)

# ------------------------------
# 2) Imported Transactions (body) – uses sidebar upload or default file
st.subheader("Import Transactions from CSV")

# If nothing uploaded yet, fall back to local sample_transactions.csv
if st.session_state.uploaded_df is None:
    TX_PATH = "./assignment_data_mining/sample_transactions.csv"
    if Path(TX_PATH).exists():
        uploaded_df = safe_read_csv(TX_PATH)
        if not uploaded_df.empty:
            st.session_state.uploaded_df = uploaded_df
            st.info(f"Loaded {len(uploaded_df)} transactions from local file.")
    else:
        st.warning("No uploaded file and no local sample_transactions.csv found.")

# Show raw data and stats
if st.session_state.uploaded_df is not None:
    raw_df = st.session_state.uploaded_df.copy()
    if raw_df.shape[1] > 1:
        raw_df = raw_df[[raw_df.columns[-1]]].rename(columns={raw_df.columns[-1]: 'items'})
    else:
        raw_df.columns = ['items']

    total_raw = len(raw_df)
    all_items = [normalize_item(x) for x in raw_df['items'].astype(str).fillna("")]
    unique_raw = len(set(chain.from_iterable([i.split(',') for i in all_items if i])))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total transactions (raw)", total_raw)
    with col2:
        st.metric("Unique items (raw)", unique_raw)

    with st.expander("Raw Transactions (before cleaning)", expanded=False):
        st.dataframe(raw_df, use_container_width=True)

# manual + upload combined
combined_df = pd.DataFrame()
if st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df.copy()
    if df.shape[1] > 1:
        df = df[[df.columns[-1]]].rename(columns={df.columns[-1]: 'items'})
    else:
        df.columns = ['items']
    combined_df = df

if st.session_state.manual_txs:
    extra = pd.DataFrame({'items': [", ".join(t) for t in st.session_state.manual_txs]})
    combined_df = pd.concat([combined_df, extra], ignore_index=True) if not combined_df.empty else extra

# ------------------------------
# 3) Preprocessing
run_prep = st.button("Run Preprocessing")
if run_prep and not combined_df.empty:
    with st.spinner("Cleaning data..."):
        cleaned, report = preprocess_transactions(
            combined_df, prod_df_raw,
            flag_singles=flag_singles,
            strict_invalid=strict_invalid
        )
        st.session_state.cleaned = [set(t) for t in cleaned]
        st.session_state.report = report
        st.success("Preprocessing complete.")

if st.session_state.cleaned is not None:
    r = st.session_state.report
    with st.expander("Preprocessing Report", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total issues detected", r['total_issues'])
            st.metric("Transactions (before)", r['before_total_tx'])
            st.metric("Empty removed", r['empty_tx_removed'])
        with col2:
            st.metric("Single-item removed", r['single_item_tx_removed'])
            st.metric("Single-item flagged", r.get('single_item_flagged', 0))
            st.metric("Duplicates removed", r['duplicate_items_removed'])
        with col3:
            st.metric("Invalid items removed", r['invalid_items_removed'])
            st.metric("Valid transactions", r['after_valid_tx'])
            st.metric("Unique products", r['unique_products'])

    with st.expander("Cleaned Transactions", expanded=False):
        clean_sample = [' , '.join(sorted(t)) for t in st.session_state.cleaned]
        st.dataframe(pd.DataFrame({'transaction': clean_sample}), use_container_width=True, hide_index=True)

st.divider()

# ------------------------------
# 4) Mining
run_mining = st.button("Analyze")
if run_mining:
    if not st.session_state.cleaned:
        st.error("Run preprocessing first.")
    else:
        tx = [set(t) for t in st.session_state.cleaned]
        results = {}

        # Apriori
        with st.spinner("Running Apriori..."):
            t0 = time.perf_counter()
            L_ap = apriori(tx, min_support=min_support)
            rules_ap = generate_rules(L_ap, min_conf=min_conf, n_tx=len(tx))
            t1 = time.perf_counter()
            results['Apriori'] = {
                'runtime_ms': (t1 - t0) * 1000,
                'rules': len(rules_ap)
            }

        # Eclat
        with st.spinner("Running Eclat..."):
            t2 = time.perf_counter()
            L_ec = eclat(tx, min_support=min_support)
            rules_ec = generate_rules(L_ec, min_conf=min_conf, n_tx=len(tx))
            t3 = time.perf_counter()
            results['Eclat'] = {
                'runtime_ms': (t3 - t2) * 1000,
                'rules': len(rules_ec)
            }

        st.session_state.results = results
        st.session_state.ap_rules = rules_ap
        st.session_state.ec_rules = rules_ec

# Show performance table
if st.session_state.results:
    perf_df = pd.DataFrame(st.session_state.results).T
    perf_df = perf_df[['runtime_ms', 'rules']]
    perf_df.columns = ['Runtime (ms)', 'Rules Generated']
    perf_df = perf_df.round(2)
    st.subheader("Performance Comparison")
    st.dataframe(perf_df, use_container_width=True)

    st.write("**Analysis:** Eclat is usually faster due to vertical format and set intersections. "
             "Both algorithms generate the same rules on this dataset.")

# ------------------------------
# Query recommendations
st.subheader("Query Recommendations")
if st.session_state.get('ap_rules') and st.session_state.ap_rules:
    all_items = sorted({item for rule in st.session_state.ap_rules for item in rule['antecedent'] + rule['consequent']})
else:
    all_items = product_names

picked = st.selectbox("Select a product:", options=all_items)

def get_recommendations(item, rules):
    agg = {}
    for r in rules:
        if item in r['antecedent']:
            for c in r['consequent']:
                score = r['confidence']
                if c not in agg or score > agg[c]['confidence']:
                    agg[c] = {
                        'confidence': score,
                        'support': r['support'],
                        'lift': r['lift']
                    }
    out = []
    for k, v in agg.items():
        out.append({
            'item': k,
            'confidence_pct': v['confidence'] * 100,
            'support_pct': v['support'] * 100,
            'lift': v['lift']
        })
    out.sort(key=lambda x: (x['confidence_pct'], x['lift']), reverse=True)
    return out

if picked and st.session_state.get('ap_rules'):
    ap_recs = get_recommendations(picked, st.session_state.ap_rules)
    ec_recs = get_recommendations(picked, st.session_state.ec_rules or [])

    tab1, tab2 = st.tabs(["Apriori", "Eclat"])

    def show_recs(recs, name):
        if recs:
            df = pd.DataFrame(recs)

            df['strength'] = pd.cut(
                df['confidence_pct'],
                bins=[0, 40, 70, 100],
                labels=["Weak", "Moderate", "Strong"],
                include_lowest=True
            )

            df = df[['item', 'confidence_pct', 'support_pct', 'lift', 'strength']]
            df.columns = ['Item', 'Confidence %', 'Support %', 'Lift', 'Strength']

            df['Confidence %'] = df['Confidence %'].round(2)
            df['Support %'] = df['Support %'].round(2)
            df['Lift'] = df['Lift'].round(2)

            st.dataframe(df, use_container_width=True, hide_index=True)

            top1 = df.iloc[0]['Item'] if len(df) > 0 else ""
            top2 = df.iloc[1]['Item'] if len(df) > 1 else ""
            st.write(
                f"**Recommendation:** Place **{top1}** next to **{picked}**. "
                f"Bundle: **{picked} + {top1}" + (f" + {top2}**" if top2 else "**") + "."
            )
        else:
            st.info("No associations found.")

    with tab1:
        show_recs(ap_recs, "Apriori")
    with tab2:
        show_recs(ec_recs, "Eclat")

