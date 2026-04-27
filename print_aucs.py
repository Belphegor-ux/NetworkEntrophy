import json

with open("dashboard_app/dashboard_data.json") as f:
    data = json.load(f)

for ds in data:
    print(f"\n{ds}:")
    for mode in ["Static", "Iterative"]:
        print(f"  {mode}:")
        for metric, vals in data[ds].get(mode, {}).items():
            print(f"    {metric}: {vals.get('auc')}")
