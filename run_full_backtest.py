import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import get_mega645_numbers
from models.vulnerability_scanner import VulnerabilityScanner
from models.exploit_engine import ExploitEngine

def main():
    print("Đang tải dữ liệu lịch sử...")
    data = get_mega645_numbers()
    if not data:
        print("Lỗi tải dữ liệu.")
        return
        
    total_draws = len(data)
    print(f"Tổng số kỳ đã tải: {total_draws}")
    
    match_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    engine = ExploitEngine(45, 6)
    
    start_time = time.time()
    
    print("Bắt đầu chạy Backtest từ kỳ 100 đến kỳ hiện tại...")
    for i in range(100, total_draws):
        train_data = data[:i]
        actual = set(data[i])
        
        # Mạo danh 1 bias để trigger Exploit Engine (vì trọng số vật lý đã hardcode bên trong)
        scan_results = {'summary': {'exploitable_biases': [{'type': 'frequency_hot', 'numbers': [], 'strength': 1.0}]}} 
        
        exploit = engine.exploit(train_data, scan_results, n_sets=1)
        
        if exploit['predictions']:
            pred = set(exploit['predictions'][0]['numbers'])
            matches = len(pred & actual)
            match_counts[matches] += 1
        else:
            match_counts[0] += 1
            
        if i % 200 == 0:
            print(f"Đã test xong {i}/{total_draws} kỳ...")
            
    test_size = total_draws - 100
    print("\n=========================================================")
    print("   KẾT QUẢ THỰC TẾ TRÊN TOÀN BỘ LỊCH SỬ (MEGA 6/45)      ")
    print("   Áp dụng: Quantum Forcing + Entropy Compression        ")
    print("=========================================================")
    print(f"Tổng số kỳ đã test: {test_size} kỳ")
    print(f"🎯 Trúng 6/6 (Jackpot) : {match_counts[6]} lần")
    print(f"🥇 Trúng 5/6 (Giải Nhất): {match_counts[5]} lần")
    print(f"🥈 Trúng 4/6 (Giải Nhì) : {match_counts[4]} lần")
    print(f"🥉 Trúng 3/6 (Giải Ba)  : {match_counts[3]} lần")
    print(f"❌ Trượt (0-2 số)       : {match_counts[0] + match_counts[1] + match_counts[2]} lần")
    
    cost = test_size * 10000
    revenue = match_counts[3] * 30000 + match_counts[4] * 300000 + match_counts[5] * 10000000 + match_counts[6] * 15000000000
    
    print("\n--- HIỆU QUẢ ĐẦU TƯ (Mỗi kỳ mua 1 vé 10.000 VNĐ) ---")
    print(f"Tổng tiền vốn bỏ ra   : {cost:,} VNĐ")
    print(f"Tổng tiền trúng thưởng: {revenue:,} VNĐ (Ước tính Jackpot = 15 Tỷ)")
    if cost > 0:
        roi = (revenue - cost) / cost * 100
        print(f"Tỷ suất sinh lời (ROI) : {roi:.2f}%")
    print(f"Thời gian test: {time.time() - start_time:.2f} giây")

if __name__ == '__main__':
    main()
