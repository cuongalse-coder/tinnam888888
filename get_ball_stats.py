import sqlite3
import pandas as pd

def analyze_balls(db_path, table_name, max_ball):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    print(f"Columns for {table_name}: {df.columns.tolist()}")
    
    all_numbers = []
    for col in df.columns:
        if 'num' in col.lower() or col in ['n1','n2','n3','n4','n5','n6'] or col in ['b1','b2','b3','b4','b5','b6']:
            all_numbers.extend(df[col].dropna().tolist())
            
    if not all_numbers:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not 'id' in col.lower() and not 'kỳ' in col.lower() and not 'draw' in col.lower() and not 'jackpot' in col.lower() and not 'prize' in col.lower() and not 'tien' in col.lower() and not 'ngay' in col.lower() and not 'date' in col.lower():
                if df[col].max() <= max_ball and df[col].min() >= 1:
                    all_numbers.extend(df[col].dropna().tolist())

    from collections import Counter
    counts = Counter([int(x) for x in all_numbers if isinstance(x, (int, float)) and x > 0])
    
    total_draws = len(df)
    total_balls = sum(counts.values())
    
    res = []
    for num in range(1, max_ball + 1):
        c = counts.get(num, 0)
        rate = c / total_draws * 100 if total_draws > 0 else 0
        res.append((num, c, rate))
        
    res.sort(key=lambda x: x[1], reverse=True)
    
    with open(f"stats_{table_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"--- THỐNG KÊ {table_name.upper()} ({total_draws} KỲ) ---\n")
        f.write("Trung bình lý thuyết mỗi quả bóng: {:.2f}%\n".format(6/max_ball * 100))
        f.write("Sắp xếp theo thứ tự xuất hiện nhiều nhất (Bóng nhẹ nhất -> nặng nhất)\n\n")
        for num, c, rate in res:
            f.write(f"Bóng {num:02d}: Ra {c} lần (Tỷ lệ {rate:.2f}% / kỳ)\n")
    print(f"Done {table_name}")

analyze_balls('data/tinnam.db', 'mega645', 45)
analyze_balls('data/tinnam.db', 'power655', 55)
