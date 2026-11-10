import torch

data = torch.load("graph_pyg.pt", weights_only=False)

print("--- PYTORCH GEOMETRIC DATA ÖZETİ ---")
print(data)

print("\n--- DETAYLI TENSÖR BİLGİLERİ ---")
print(f"Düğüm Sayısı (Nodes): {data.num_nodes}")
print(f"Kenar Sayısı (Edges): {data.num_edges}")
print(f"Düğüm Özellik Matrisi (x): {data.x.shape}")
print(f"Kenar Bağlantı Matrisi (edge_index): {data.edge_index.shape}")

if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    print(f"Kenar Ağırlıkları (edge_attr) Boyutu: {data.edge_attr.shape}")
    print(f"Kenar Ağırlığı Min / Max / Ort: {data.edge_attr.min():.4f} / {data.edge_attr.max():.4f} / {data.edge_attr.mean():.4f}")

if hasattr(data, 'node_weight') and data.node_weight is not None:
    print(f"Düğüm Ağırlıkları (node_weight) Boyutu: {data.node_weight.shape}")
    print(f"Düğüm Ağırlığı Min / Max / Ort: {data.node_weight.min():.4f} / {data.node_weight.max():.4f} / {data.node_weight.mean():.4f}")

if hasattr(data, 'y') and data.y is not None:
    print(f"Aile Etiketleri (y) Boyutu: {data.y.shape}")

if hasattr(data, 'y_binary') and data.y_binary is not None:
    print(f"İkili Etiketler (y_binary) Boyutu: {data.y_binary.shape}")

print("\n--- ÖRNEK İLK 3 KENAR BAĞLANTISI ---")
for i in range(min(3, data.num_edges)):
    src = data.edge_index[0, i].item()
    dst = data.edge_index[1, i].item()
    weight = data.edge_attr[i].item() if hasattr(data, 'edge_attr') and data.edge_attr is not None else "N/A"
    print(f"Kenar {i}: Düğüm {src} -> Düğüm {dst} | Ağırlık: {weight:.4f}")