# streamlit_graph_analysis_full.py
"""
Connected Graph Visualizer & Analyzer (enhanced)
- interactive PyVis embedded view
- Plotly interactive view
- exports: CSV, PNG, GraphML
- performance improvements (approx connectivity, toggles)
- multiple clustering methods + quality plots
"""
import streamlit as st
st.set_page_config(layout="wide", page_title="Graph Visualizer & Analyzer")

import networkx as nx
import numpy as np
import pandas as pd
import io
import base64
import sys
import time
from pyvis.network import Network
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import spectral_embedding
import plotly.graph_objects as go
import community as community_louvain  # python-louvain
from networkx.algorithms.connectivity import node_connectivity as nx_node_connectivity, edge_connectivity as nx_edge_connectivity
from networkx.algorithms import approximation as nx_approx
from networkx.algorithms import community as nx_community
import warnings
warnings.filterwarnings("ignore")

# ---------- Sidebar control ----------
st.sidebar.title("Connected Graph parameters")
default_n = 50
n = st.sidebar.number_input("Number of nodes (n)", min_value=5, max_value=1500, value=default_n, step=1)
m = st.sidebar.number_input("Number of edges (m) — must be > n", min_value=n+1, max_value=10**7, value=n*3, step=1)
family = st.sidebar.selectbox("Graph family", ["random (Erdős–Rényi)", "small-world (Watts–Strogatz)",
                                              "d-regular (random)", "complete", "generalized Petersen"])

# family-specific
if family == "small-world (Watts–Strogatz)":
    ws_k = st.sidebar.slider("WS: k (even, nearest neighbors)", min_value=2, max_value=min(50, n-1), value=4, step=1)
    if ws_k % 2 != 0: ws_k += 1
    ws_p = st.sidebar.slider("WS: rewiring p", 0.0, 1.0, 0.15, 0.01)
if family == "d-regular (random)":
    d = st.sidebar.number_input("d for d-regular graph", min_value=0, max_value=n-1, value=3, step=1)
if family == "generalized Petersen":
    gp_k = st.sidebar.number_input("Petersen parameter k (1 ≤ k < n/2)", min_value=1, max_value=max(1,(n//2)-1), value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Clustering & analysis options")
clust_method = st.sidebar.selectbox("Clustering method", ["Louvain (modularity)", "SpectralClustering (sklearn)", "Spectral embedding + KMeans"])
do_clustering = st.sidebar.checkbox("Run clustering", value=False)
k_clusters = st.sidebar.slider("Number of clusters (k, for applicable methods)", 2, 10, 3)

#st.sidebar.markdown("---")
#st.sidebar.header("Performance / advanced")
exact_connectivity = st.sidebar.checkbox("Compute exact node/edge connectivity (slow for large n)", value=False)
large_n_threshold = st.sidebar.number_input("Large n threshold (warn for n ≥)", min_value=50, max_value=5000, value=500, step=50)
show_plotly = st.sidebar.checkbox("Show Plotly interactive view", value=True)

# ---------- Utility functions ----------
def clip_m(n_nodes, requested_m):
    max_edges = n_nodes*(n_nodes-1)//2
    min_edges = n_nodes - 1
    if requested_m < min_edges:
        st.warning(f"m too small for connectivity; raising to {min_edges}.")
        return min_edges
    if requested_m > max_edges:
        st.warning(f"m > max simple edges ({max_edges}); clipping to {max_edges}.")
        return max_edges
    return requested_m

def safe_write_graphml(G):
    bio = io.BytesIO()
    try:
        nx.write_graphml(G, bio)
        bio.seek(0)
        return bio
    except Exception as e:
        st.error(f"GraphML export failed: {e}")
        return None

def generate_connected_graph(n, m, family):
    rng = np.random.default_rng()
    m = clip_m(n, m)
    if family == "random (Erdős–Rényi)":
        p = (2.0*m) / (n*(n-1))
        for attempt in range(200):
            H = nx.gnp_random_graph(n, p, seed=int(rng.integers(1e9)))
            if nx.is_connected(H):
                G = H.copy()
                break
        else:
            # fallback: build spanning tree + random edges
            G = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
            add_random_edges(G, m, rng)
    elif family == "small-world (Watts–Strogatz)":
        k_local = ws_k if 'ws_k' in globals() else 4
        p_local = ws_p if 'ws_p' in globals() else 0.15
        G = nx.connected_watts_strogatz_graph(n, k_local, p_local, seed=int(rng.integers(1e9)))
        add_random_edges(G, m, rng)
    elif family == "d-regular (random)":
        d_local = d if 'd' in globals() else 3
        # ensure valid
        if d_local >= n:
            d_local = n-1
        if (d_local * n) % 2 != 0:
            d_local -= 1 if d_local>0 else 0
        try:
            G = nx.random_regular_graph(d_local, n, seed=int(rng.integers(1e9)))
            add_random_edges(G, m, rng)
        except Exception:
            G = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
            add_random_edges(G, m, rng)
    elif family == "complete":
        G = nx.complete_graph(n)
    elif family == "generalized Petersen":
        k_local = gp_k if 'gp_k' in globals() else 2
        try:
            G = nx.generators.small.generalized_petersen_graph(n, k_local)
            add_random_edges(G, m, rng)
        except Exception:
            G = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
            add_random_edges(G, m, rng)
    else:
        G = nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))
        add_random_edges(G, m, rng)

    # ensure connected by linking components if necessary
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps)-1):
            a = next(iter(comps[i])); b = next(iter(comps[i+1])); G.add_edge(a,b)
    return G

def add_random_edges(G, target_m, rng):
    """Add random edges until G has target_m edges (avoid duplicates)."""
    n = G.number_of_nodes()
    max_edges = n*(n-1)//2
    target_m = min(max(target_m, n-1), max_edges)
    while G.number_of_edges() < target_m:
        u = rng.integers(0, n); v = rng.integers(0, n)
        if u == v or G.has_edge(u,v): continue
        G.add_edge(int(u), int(v))

def nx_safe_diameter(G):
    try:
        return nx.diameter(G)
    except Exception:
        return None

# ---------- Generate graph & warn about large n ----------
if n >= large_n_threshold:
    st.warning(f"n = {n} is ≥ {large_n_threshold}. Some computations (exact connectivity, girth, exact diameter) may be slow or memory-heavy. You can toggle exact computations in sidebar.")

with st.spinner("Generating graph..."):
    G = generate_connected_graph(n, m, family)

# ---------- Basic metrics ----------
degrees = np.array([d for _, d in G.degree()])
metrics = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "density": np.round(nx.density(G),4),
    "min_degree": int(degrees.min()),
    "max_degree": int(degrees.max()),
    "avg_degree": np.round(float(degrees.mean()),4),
    "diameter": nx_safe_diameter(G),
}

# girth: attempt to compute shortest cycle length; use approximation if large
def compute_girth(G, cutoff=1000000):
    # naive: BFS per node until shortest cycle found
    n = G.number_of_nodes()
    if n > 2000:
        # skip expensive exact girth for huge graphs
        return "skipped (n large)"
    try:
        return nx.girth(G)
    except Exception:
        # fallback: brute force BFS per node
        best = np.inf
        for s in G.nodes():
            visited = {s: 0}
            parent = {s: None}
            q = [s]
            qi = 0
            while qi < len(q):
                u = q[qi]; qi+=1
                for v in G.neighbors(u):
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v:
                        # found cycle length
                        length = visited[u] + visited[v] + 1
                        best = min(best, length)
                        if best == 3:
                            return 3
        return int(best) if np.isfinite(best) else "acyclic"

metrics["girth"] = compute_girth(G)

# connectivity: either exact or approximate depending on toggle & size
if exact_connectivity or n < 500:
    try:
        metrics["node_connectivity"] = nx_node_connectivity(G)
    except Exception as e:
        metrics["node_connectivity"] = f"error: {e}"
    try:
        metrics["edge_connectivity"] = nx_edge_connectivity(G)
    except Exception as e:
        metrics["edge_connectivity"] = f"error: {e}"
else:
    # use approximation / lower bound (faster)
    try:
        # networkx approximation lower bound
        metrics["node_connectivity"] = nx_approx.node_connectivity(G)
    except Exception:
        metrics["node_connectivity"] = "approximation_unavailable"
    metrics["edge_connectivity"] = "skipped (approx not provided)"

metrics["avg_clustering"] = np.round(nx.average_clustering(G),4)
try:
    metrics["local_efficiency"] = np.round(nx.local_efficiency(G),4)
    metrics["global_efficiency"] = np.round(nx.global_efficiency(G),4)
except Exception:
    metrics["local_efficiency"] = "error"
    metrics["global_efficiency"] = "error"

# algebraic connectivity (normalized)
try:
    metrics["normalized_algebraic_connectivity"] = np.round(nx.algebraic_connectivity(G, normalized=True),4)
except Exception:
    metrics["normalized_algebraic_connectivity"] = "error/computation-skipped"

# planarity & connectedness
try:
    is_planar, _ = nx.check_planarity(G)
    metrics["is_planar"] = bool(is_planar)
except Exception:
    metrics["is_planar"] = "error"
metrics["is_connected"] = nx.is_connected(G)

# modularity via Louvain (fast)
try:
    partition = community_louvain.best_partition(G)
    # convert to communities
    comms = {}
    for node, cid in partition.items():
        comms.setdefault(cid, []).append(node)
    comms_list = list(comms.values())
    modularity = np.round(community_louvain.modularity(partition, G),4)
    metrics["modularity"] = modularity
    metrics["n_communities_louvain"] = len(comms_list)
except Exception:
    metrics["modularity"] = "error"
    metrics["n_communities_louvain"] = "error"

# ---------- UI display ----------
st.header("Graph Analyzer",divider="rainbow")
st.markdown("The following interface, takes as input simple graph specifications such as number of nodes, suggested edges, and family type, and generates a connected graph accordingly.It then computes various graph metrics, provides interactive visualizations and clustering, and allows for exporting the graph and its metrics.")
st.badge("Developed by: Al Yazdani")
left, right = st.columns([1,2])

with left:
    st.subheader("Graph metrics",divider="rainbow")
    #for k,v in metrics.items():
        #st.write(f"**{k}**: {v}")
        
    st.dataframe(metrics,height="content")
    st.markdown("---")
    st.subheader("Export",divider="rainbow")
    # metrics -> CSV
    df_metrics = pd.DataFrame(list(metrics.items()), columns=["metric","value"])
    csv_bytes = df_metrics.to_csv(index=False).encode()
    st.download_button("Download metrics (CSV)", data=csv_bytes, file_name="graph_metrics.csv", mime="text/csv")

    # graphml
    bio = safe_write_graphml(G)
    if bio:
        st.download_button("Download graph (GraphML)", data=bio, file_name="graph.graphml", mime="application/graphml+xml")

    # adjacency as PNG: build plotly fig and download as PNG via static image (Streamlit's download_button requires bytes)
    # We'll create matplotlib figure for easy PNG download:
    fig_png = plt.figure(figsize=(7,7))
    #pos = nx.spring_layout(G, seed=42) if n<=300 else nx.spectral_layout(G)
    pos= nx.spiral_layout(G) 
    nx.draw(G, pos=pos, node_size=40 if n>200 else 120, with_labels=False)
    buf = io.BytesIO(); fig_png.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    st.download_button("Download network image (PNG)", data=buf, file_name="graph.png", mime="image/png")
    plt.close(fig_png)

with right:
    st.subheader("Graph visualization",divider="rainbow")
    
    # Plotly view (lighter, hover)
    if show_plotly:
        try:
            #pos = nx.spring_layout(G, seed=42) if n<=300 else nx.spectral_layout(G)
            pos= nx.spiral_layout(G)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.8), hoverinfo='none')
            node_x=[]; node_y=[]; node_text=[]
            for node in G.nodes():
                x,y = pos[node]
                node_x.append(x); node_y.append(y)
                node_text.append(f"node {node} — deg {G.degree(node)}")
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                    text=node_text, marker=dict(size=10, color=[G.degree(n) for n in G.nodes()], showscale=True))
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(#title="Plotly network view (hover nodes)", 
                                             showlegend=False,
                                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                             height=650))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly network failed: {e}")

# ---------- Degree distribution / adjacency / Laplacian spectrum ----------
st.markdown("---")
st.subheader("Degree distribution / Adjacency / Laplacian spectrum")

col_a, col_b, col_c = st.columns(3)
with col_a:
    degs = [d for _,d in G.degree()]
    fig1 = plt.figure()
    plt.hist(degs, bins=range(min(degs), max(degs)+2))
    plt.xlabel("degree"); plt.ylabel("count"); plt.title("Degree Distribution")
    st.pyplot(fig1)
    plt.close(fig1)
with col_b:
    A = nx.to_numpy_array(G)
    fig2 = plt.figure(figsize=(4,4))
    plt.imshow(A, interpolation='nearest')
    plt.title("Adjacency matrix image")
    st.pyplot(fig2)
    plt.close(fig2)
with col_c:
    L = nx.laplacian_matrix(G).astype(float).todense()
    eigs = np.sort(np.real(np.linalg.eigvals(L)))
    fig3 = plt.figure()
    plt.plot(eigs, 'o-'); plt.title("Laplacian spectrum (sorted)")
    st.pyplot(fig3)
    plt.close(fig3)

# ---------- Clustering ----------
if do_clustering:
    st.markdown("---")
    st.subheader("Clustering and cluster quality")

    A = nx.to_numpy_array(G)
    labels = None
    cluster_ok = True
    if clust_method == "SpectralClustering (sklearn)":
        try:
            sc = SpectralClustering(n_clusters=k_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42, n_init=10)
            labels = sc.fit_predict(A)
        except Exception as e:
            st.error(f"SpectralClustering failed: {e}")
            cluster_ok = False
    elif clust_method == "Spectral embedding + KMeans":
        try:
            emb = spectral_embedding(A, n_components=min(50, max(2, k_clusters+2)), random_state=42)
            km = KMeans(n_clusters=k_clusters, n_init=20, random_state=42)
            labels = km.fit_predict(emb)
        except Exception as e:
            st.error(f"Embedding+KMeans failed: {e}")
            cluster_ok = False
    elif clust_method == "Louvain (modularity)":
        try:
            part = community_louvain.best_partition(G)
            # map partition labels to contiguous [0..]
            unique = sorted(set(part.values()))
            mapping = {u:i for i,u in enumerate(unique)}
            labels = np.array([mapping[part[node]] for node in sorted(G.nodes())])
            # if user specified k_clusters, we keep Louvain's k
        except Exception as e:
            st.error(f"Louvain failed: {e}")
            cluster_ok = False

    if cluster_ok and labels is not None:
        st.write(f"**The {clust_method} method found {len(set(labels))} clusters.**")
        # cluster metrics using spectral embedding (if available) else adjacency rows
        try:
            emb_for_metrics = spectral_embedding(A, n_components=min(50, max(2, len(set(labels))+2)), random_state=42)
        except Exception:
            emb_for_metrics = A
        cm_sil = silhouette_score(emb_for_metrics, labels) if len(set(labels))>1 else "N/A"
        cm_db = davies_bouldin_score(emb_for_metrics, labels) if len(set(labels))>1 else "N/A"
        cm_ch = calinski_harabasz_score(emb_for_metrics, labels) if len(set(labels))>1 else "N/A"
        
        clustering_metrics = pd.DataFrame({"Metric": ["Silhouette score", "Davies-Bouldin index", "Calinski-Harabasz index"],
                      "Value": [cm_sil, cm_db, cm_ch]}).set_index("Metric")
        
        st.dataframe(clustering_metrics, height="content")
        #st.write({"silhouette":cm_sil, "davies_bouldin":cm_db, "calinski_harabasz":cm_ch})

        # show elbow (inertia) if KMeans used
        if clust_method == "Spectral embedding + KMeans":
            st.write("Elbow (inertia) plot for k=1..10 on embedding")
            inertias = []
            K_range = range(1, min(11, n))
            for K in K_range:
                km = KMeans(n_clusters=K, n_init=10, random_state=42)
                km.fit(emb)
                inertias.append(km.inertia_)
            fig_el = plt.figure()
            plt.plot(list(K_range), inertias, '-o')
            plt.xlabel("k"); plt.ylabel("inertia"); plt.title("Elbow: inertia vs k")
            st.pyplot(fig_el)
            plt.close(fig_el)

        # draw colored graph by cluster
        colormap = px_colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]
        node_colors = []
        for idx, node in enumerate(sorted(G.nodes())):
            lbl = int(labels[idx])
            node_colors.append(colormap[lbl % len(colormap)])
        # plotly colored graph
        pos = nx.spring_layout(G, seed=42) if n<=300 else nx.spectral_layout(G)
        edge_x=[]; edge_y=[]
        for e in G.edges():
            x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.8), hoverinfo='none')
        node_x=[]; node_y=[]
        node_text=[]
        for node in sorted(G.nodes()):
            x,y = pos[node]; node_x.append(x); node_y.append(y)
            node_text.append(f"node {node} — cluster {labels[list(sorted(G.nodes())).index(node)]}")
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                text=node_text, marker=dict(size=10, color=node_colors))
        figc = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Clustered network", showlegend=False,
                                                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=650))
        st.plotly_chart(figc, use_container_width=True)

    else:
        st.info("No clustering result available.")

st.markdown("---")
#st.caption("References: PyVis examples & Streamlit embedding, Plotly network graphs, NetworkX read/write GraphML docs, NetworkX approximate connectivity and performance notes, scikit-learn clustering docs.")
