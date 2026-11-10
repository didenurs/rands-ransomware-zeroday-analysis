import torch
import numpy as np
from pyvis.network import Network

print("1. Graf verisi yükleniyor...")
data = torch.load("graph_pyg.pt", weights_only=False)

# 12 sınıf için ayırt edici renk paleti (Sınıf 0: Benign / Yeşil, diğerleri aileler)
COLOR_PALETTE = [
    "#2ecc71",  # 0: Benign (Yeşil)
    "#e74c3c",  # 1: Kırmızı
    "#3498db",  # 2: Mavi
    "#9b59b6",  # 3: Mor
    "#e67e22",  # 4: Turuncu
    "#1abc9c",  # 5: Turkuaz
    "#f1c40f",  # 6: Sarı
    "#e84393",  # 7: Pembe
    "#00cec9",  # 8: Açık Mavi
    "#6c5ce7",  # 9: Koyu Mor
    "#d63031",  # 10: Koyu Kırmızı
    "#fdcb6e"   # 11: Açık Turuncu
]

# PyVis Ağı Başlatma
net = Network(height="900px", width="100%", bgcolor="#1e272e", font_color="white")
net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=95, spring_strength=0.04)

print("2. Düğümler ekleniyor...")
# Düğümleri ekleme
for node_id in range(data.num_nodes):
    family_label = int(data.y[node_id].item())
    is_binary = int(data.y_binary[node_id].item())
    weight = float(data.node_weight[node_id].item()) if hasattr(data, 'node_weight') else 0.5
    
    color = COLOR_PALETTE[family_label % len(COLOR_PALETTE)]
    size = 6 + (weight * 14)  # Ağırlığa göre düğüm büyüklüğü
    
    label_info = "Benign" if is_binary == 0 else f"Family {family_label}"
    title_tooltip = f"Düğüm ID: {node_id}<br>Tür: {label_info}<br>Aktivite Ağırlığı: {weight:.4f}"
    
    net.add_node(node_id, label=str(node_id), title=title_tooltip, color=color, size=size)

print("3. Kenarlar ekleniyor...")
# Tarayıcı performansını korumak için en güçlü kenarları ekleme
# Jaccard benzerliği yüksek olan (ör. >= 0.70) kenarları filtreleyerek görseli netleştiriyoruz
edge_index = data.edge_index.numpy()
edge_attr = data.edge_attr.numpy() if hasattr(data, 'edge_attr') else np.ones(data.num_edges)

similarity_threshold = 0.70  # Sadece belirgin bağlantıları göster (tümünü görmek isterseniz 0.0 yapabilirsiniz)
added_edges = 0

for i in range(data.num_edges):
    weight = float(edge_attr[i])
    if weight >= similarity_threshold:
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        net.add_edge(src, dst, value=weight, title=f"Jaccard Benzerliği: {weight:.3f}", color="rgba(200, 200, 200, 0.15)")
        added_edges += 1

print(f"-> Eşik değeri (>={similarity_threshold}) üzerindeki {added_edges} kenar çizildi.")

# HTML olarak kaydet
OUTPUT_FILE = "ransomware_graph.html"
net.write_html(OUTPUT_FILE)
print(f"\n--- Görselleştirme Hazır: '{OUTPUT_FILE}' dosyasını tarayıcınızda açabilirsiniz! ---")