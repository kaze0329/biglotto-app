import os
import platform
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 設定 Matplotlib 使用 TkAgg 後端並自動適配繁體中文字型
matplotlib.use("TkAgg")
sys_name = platform.system()
if sys_name == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
elif sys_name == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


class BigLottoGUI:
    def __init__(self, root: tk.Tk, csv_path: str = "taiwan_biglotto_2004_2026.csv"):
        self.root = root
        self.csv_path = csv_path
        self.root.title("台灣大樂透智慧預測與視覺化走勢分析系統 (目標：六獎 3+1 / 五獎 4+0 以上)")
        self.root.geometry("1460x880")
        self.root.minsize(1280, 780)
        self.root.configure(bg="#f0f2f5")

        self.num_cols = ["獎號1", "獎號2", "獎號3", "獎號4", "獎號5", "獎號6"]
        self.df = self._load_or_init_data()

        # 儲存最新運算結果供匯出使用
        self.latest_result = {}

        self._setup_styles()
        self._build_ui()
        self.run_analysis_and_update()

    def _load_or_init_data(self) -> pd.DataFrame:
        """載入歷史 CSV 檔案，若不存在則建立空白結構"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            for col in self.num_cols + ["特別號"]:
                df[col] = df[col].astype(int)
            return df
        else:
            cols = ["開獎日期"] + self.num_cols + ["特別號"]
            return pd.DataFrame(columns=cols)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabelframe", background="#ffffff", font=("Microsoft JhengHei", 10, "bold"))
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#2c3e50", font=("Microsoft JhengHei", 10, "bold"))
        style.configure("TLabel", background="#ffffff", foreground="#2c3e50", font=("Microsoft JhengHei", 9))
        style.configure("Primary.TButton", font=("Microsoft JhengHei", 10, "bold"), foreground="white", background="#2980b9", padding=6)
        style.map("Primary.TButton", background=[("active", "#1f6391")])
        style.configure("Success.TButton", font=("Microsoft JhengHei", 10, "bold"), foreground="white", background="#27ae60", padding=6)
        style.map("Success.TButton", background=[("active", "#1e8449")])
        style.configure("Treeview.Heading", font=("Microsoft JhengHei", 9, "bold"), background="#dfe6e9", foreground="#2d3436")
        style.configure("Treeview", font=("Consolas", 10), rowheight=25)

    def _build_ui(self):
        # 主容器：左右分欄
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_paned, width=540)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=2)
        main_paned.add(right_panel, weight=5)

        # ==================== 左側面板 ====================
        # 1. 手動輸入歷史獎號區
        input_frame = ttk.LabelFrame(left_panel, text=" 📌 手動新增最新開獎獎號 (自動儲存至 CSV) ", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(input_frame)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="開獎日期 (YYYY/MM/DD):").pack(side=tk.LEFT)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y/%m/%d"))
        ttk.Entry(row1, textvariable=self.date_var, width=14, font=("Consolas", 10)).pack(side=tk.LEFT, padx=6)

        ttk.Button(row1, text="📂 載入其他 CSV", command=self.load_custom_csv).pack(side=tk.RIGHT)

        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=6)
        ttk.Label(row2, text="6個一般號 (1~49):").pack(side=tk.LEFT)
        self.num_vars = [tk.StringVar() for _ in range(6)]
        for var in self.num_vars:
            ttk.Entry(row2, textvariable=var, width=4, justify="center", font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=2)

        ttk.Label(row2, text=" 特別號:", foreground="#c0392b", font=("Microsoft JhengHei", 9, "bold")).pack(side=tk.LEFT, padx=(6, 2))
        self.spec_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.spec_var, width=4, justify="center", font=("Consolas", 11, "bold"), foreground="#c0392b").pack(side=tk.LEFT, padx=2)

        btn_add = ttk.Button(input_frame, text="＋ 新增獎號並更新模型", style="Success.TButton", command=self.add_new_draw)
        btn_add.pack(fill=tk.X, pady=(4, 0))

        # 2. 預測參數與操作按鈕區
        param_frame = ttk.LabelFrame(left_panel, text=" ⚙️ 智慧縮水包牌參數設定 (目標：六獎 3+1 / 五獎 4+0) ", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 8))

        p_row = ttk.Frame(param_frame)
        p_row.pack(fill=tk.X, pady=3)
        ttk.Label(p_row, text="主候選池大小 (碼):").pack(side=tk.LEFT)
        self.pool_size_var = tk.IntVar(value=12)
        ttk.Spinbox(p_row, from_=9, to=18, textvariable=self.pool_size_var, width=5, font=("Consolas", 10)).pack(side=tk.LEFT, padx=(4, 15))

        ttk.Label(p_row, text="推薦縮水注數 (注):").pack(side=tk.LEFT)
        self.ticket_count_var = tk.IntVar(value=6)
        ttk.Spinbox(p_row, from_=3, to=15, textvariable=self.ticket_count_var, width=5, font=("Consolas", 10)).pack(side=tk.LEFT, padx=4)

        b_row = ttk.Frame(param_frame)
        b_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(b_row, text="⚡ 重新執行預測與繪圖", style="Primary.TButton", command=self.run_analysis_and_update).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(b_row, text="💾 匯出推薦組合 (.txt)", command=self.export_predictions).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # 3. 預測結果顯示區
        res_frame = ttk.LabelFrame(left_panel, text=" 🎯 下一期智慧預測與貪婪縮水包牌結果 ", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.info_label = ttk.Label(res_frame, text="運算中...", font=("Microsoft JhengHei", 9, "bold"), foreground="#16a085", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X, pady=(0, 6))

        cols = ("注別", "推薦6碼組合", "特別號建議", "總和", "單雙比", "大小比")
        self.ticket_tree = ttk.Treeview(res_frame, columns=cols, show="headings", height=7)
        col_widths = [45, 175, 75, 50, 55, 55]
        for c, w in zip(cols, col_widths):
            self.ticket_tree.heading(c, text=c)
            self.ticket_tree.column(c, width=w, anchor="center")
        self.ticket_tree.pack(fill=tk.BOTH, expand=True)

        # 4. 最新歷史開獎紀錄檢視區
        hist_frame = ttk.LabelFrame(left_panel, text=" 📜 近期歷史開獎紀錄 (最新 15 期) ", padding=10)
        hist_frame.pack(fill=tk.BOTH, expand=True)

        h_cols = ("開獎日期", "一般獎號 (6碼)", "特別號", "6碼總和")
        self.hist_tree = ttk.Treeview(hist_frame, columns=h_cols, show="headings", height=6)
        h_widths = [95, 185, 60, 65]
        for c, w in zip(h_cols, h_widths):
            self.hist_tree.heading(c, text=c)
            self.hist_tree.column(c, width=w, anchor="center")

        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.hist_tree.yview)
        self.hist_tree.configure(yscroll=scrollbar.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ==================== 右側視覺化面板 ====================
        chart_frame = ttk.LabelFrame(right_panel, text=" 📊 大樂透多維度統計規律與動態走勢圖 (支援縮放與圖片儲存) ", padding=8)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(9.5, 7.2), facecolor="#f8f9fa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def add_new_draw(self):
        """驗證使用者手動輸入的獎號，寫入 CSV 並即時更新模型與圖表"""
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showerror("輸入錯誤", "請輸入開獎日期 (格式如 2026/11/03)！")
            return

        try:
            nums = [int(v.get().strip()) for v in self.num_vars]
            spec = int(self.spec_var.get().strip())
        except ValueError:
            messagebox.showerror("輸入錯誤", "請確保 6 個一般號與 1 個特別號皆為整數數字！")
            return

        all_seven = nums + [spec]
        if any(n < 1 or n > 49 for n in all_seven):
            messagebox.showerror("範圍錯誤", "所有獎號與特別號必須介於 1 到 49 之間！")
            return

        if len(set(all_seven)) != 7:
            messagebox.showerror("重複錯誤", "6 個一般號與特別號之間不可有重複的號碼！")
            return

        sorted_nums = sorted(nums)
        new_row = pd.DataFrame([{
            "開獎日期": date_str,
            "獎號1": sorted_nums[0],
            "獎號2": sorted_nums[1],
            "獎號3": sorted_nums[2],
            "獎號4": sorted_nums[3],
            "獎號5": sorted_nums[4],
            "獎號6": sorted_nums[5],
            "特別號": spec
        }])

        # 將最新一期插入 DataFrame 最上方並存回 CSV
        self.df = pd.concat([new_row, self.df], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")

        # 清空號碼輸入框
        for v in self.num_vars:
            v.set("")
        self.spec_var.set("")

        self.run_analysis_and_update()
        messagebox.showinfo("新增成功", f"已成功新增 {date_str} 開獎紀錄：\n一般號：{sorted_nums} | 特別號：{spec:02d}\n預測模型與走勢圖已完成同步更新！")

    def load_custom_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            self.df = self._load_or_init_data()
            self.run_analysis_and_update()

    def _compute_predictions(self, pool_size: int, num_tickets: int) -> dict:
        """核心複合權重評分與貪婪縮水包牌演算法"""
        main_matrix = self.df[self.num_cols].values
        spec_array = self.df["特別號"].values

        # 1. 全歷史長線頻率 (35%)
        all_counts = pd.Series(main_matrix.flatten()).value_counts()
        freq_all = np.array([all_counts.get(n, 0) for n in range(1, 50)])
        norm_all = (freq_all - freq_all.min()) / (freq_all.max() - freq_all.min() + 1e-6)

        # 2. 近 30 期短期動能 (45%)
        recent_counts = pd.Series(main_matrix[:30].flatten()).value_counts()
        freq_rec = np.array([recent_counts.get(n, 0) for n in range(1, 50)])
        norm_rec = (freq_rec - freq_rec.min()) / (freq_rec.max() - freq_rec.min() + 1e-6)

        # 3. 遺漏期數高斯回補評分 (20%，峰值設於遺漏 7 期)
        gaps = np.zeros(49)
        for n in range(1, 50):
            found = np.where((main_matrix == n).any(axis=1))[0]
            gaps[n - 1] = found[0] if len(found) > 0 else len(self.df)
        gap_score = np.exp(-((gaps - 7.0) ** 2) / (2 * 5.0 ** 2))

        # 4. 上期連莊加分 (+0.18，歷史實證 56.28% 期數含連莊號)
        last_draw_set = set(main_matrix[0])
        repeat_bonus = np.array([0.18 if n in last_draw_set else 0.0 for n in range(1, 50)])

        total_score = 0.35 * norm_all + 0.45 * norm_rec + 0.20 * gap_score + repeat_bonus

        # 特別號獨立加權積分
        spec_all = pd.Series(spec_array).value_counts()
        spec_rec = pd.Series(spec_array[:100]).value_counts()
        spec_scores = {n: spec_all.get(n, 0) * 0.6 + spec_rec.get(n, 0) * 2.5 for n in range(1, 50)}
        top_specials = [k for k, v in sorted(spec_scores.items(), key=lambda x: x[1], reverse=True)]

        ranked_nums = [int(i + 1) for i in np.argsort(total_score)[::-1]]
        candidate_pool = sorted(ranked_nums[:pool_size])

        # 黃金結構濾網
        def passes_filters(combo):
            s = sum(combo)
            if not (115 <= s <= 185):
                return False
            if not (2 <= sum(1 for n in combo if n % 2 == 1) <= 4):
                return False
            if not (2 <= sum(1 for n in combo if n >= 25) <= 4):
                return False
            if sum(1 for i in range(5) if combo[i + 1] - combo[i] == 1) > 2:
                return False
            return True

        all_possible = list(combinations(candidate_pool, 6))
        valid_combos = [c for c in all_possible if passes_filters(c)]
        if len(valid_combos) < num_tickets:
            valid_combos = all_possible

        # 近 300 期雙連碼共現矩陣
        pair_counter = Counter()
        for row in main_matrix[:300]:
            for p in combinations(sorted(row), 2):
                pair_counter[p] += 1

        # 貪婪集合覆蓋演算法 (最大化 3 碼組合覆蓋率)
        uncovered_triplets = set(combinations(candidate_pool, 3))
        total_triplets_count = max(len(uncovered_triplets), 1)
        selected_tickets = []
        remaining = valid_combos.copy()

        for _ in range(min(num_tickets, len(remaining))):
            best_c, best_g, best_syn = None, -1, -1.0
            for c in remaining:
                g = len(set(combinations(c, 3)).intersection(uncovered_triplets))
                syn = sum(total_score[n - 1] for n in c) + sum(pair_counter[p] * 0.05 for p in combinations(c, 2))
                if g > best_g or (g == best_g and syn > best_syn):
                    best_g, best_syn, best_c = g, syn, c
            selected_tickets.append(best_c)
            uncovered_triplets -= set(combinations(best_c, 3))
            remaining.remove(best_c)

        coverage_pct = (1.0 - len(uncovered_triplets) / total_triplets_count) * 100

        return {
            "total_score": total_score,
            "gaps": gaps,
            "spec_scores": spec_scores,
            "candidate_pool": candidate_pool,
            "top_specials": top_specials[:3],
            "tickets": selected_tickets,
            "coverage_pct": coverage_pct
        }

    def run_analysis_and_update(self):
        """執行預測、更新左側表格與右側四合一視覺化圖表"""
        if len(self.df) == 0:
            return

        pool_size = self.pool_size_var.get()
        num_tickets = self.ticket_count_var.get()
        res = self._compute_predictions(pool_size, num_tickets)
        self.latest_result = res

        # 1. 更新上方指標文字
        pool_str = ", ".join(f"{n:02d}" for n in res["candidate_pool"])
        spec_str = ", ".join(f"{n:02d}" for n in res["top_specials"])
        latest_date = self.df.iloc[0]["開獎日期"]
        self.info_label.config(
            text=(
                f"📊 歷史樣本數：{len(self.df)} 期 (最新：{latest_date})  |  3碼縮水覆蓋率：{res['coverage_pct']:.1f}%\n"
                f"🔥 精選 {pool_size} 碼主候選池：[{pool_str}]\n"
                f"⭐ 特別號高頻 Top 3：[{spec_str}] (建議搭配鎖定六獎 3+1)"
            )
        )

        # 2. 更新推薦注數 Treeview
        for item in self.ticket_tree.get_children():
            self.ticket_tree.delete(item)

        for idx, t in enumerate(res["tickets"], 1):
            t_str = "  ".join(f"{n:02d}" for n in t)
            sp_pick = f"{res['top_specials'][(idx - 1) % len(res['top_specials'])]:02d}"
            s_val = sum(t)
            odds = sum(1 for n in t if n % 2 == 1)
            highs = sum(1 for n in t if n >= 25)
            self.ticket_tree.insert("", tk.END, values=(f"第 {idx:02d} 注", t_str, sp_pick, s_val, f"{odds}:{6-odds}", f"{highs}:{6-highs}"))

        # 3. 更新歷史紀錄 Treeview (顯示最新 15 期)
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)

        for _, row in self.df.head(15).iterrows():
            nums = [int(row[c]) for c in self.num_cols]
            n_str = " ".join(f"{n:02d}" for n in nums)
            self.hist_tree.insert("", tk.END, values=(row["開獎日期"], n_str, f"{int(row['特別號']):02d}", sum(nums)))

        # 4. 更新右側四合一走勢圖
        self._update_charts(res)

    def _update_charts(self, res: dict):
        for ax in self.axes.flat:
            ax.clear()

        main_matrix = self.df[self.num_cols].values
        cand_set = set(res["candidate_pool"])

        # 圖 1：1~49號 綜合動能評分
        ax1 = self.axes[0, 0]
        colors1 = ["#e74c3c" if (i + 1) in cand_set else "#bdc3c7" for i in range(49)]
        ax1.bar(range(1, 50), res["total_score"], color=colors1)
        ax1.set_title(f"1~49號 綜合動能評分 (紅色：入選 {len(cand_set)} 碼主候選池)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("獎號 (1~49)", fontsize=8)
        ax1.set_ylabel("綜合權重分數", fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # 圖 2：近 50 期獎號總和走勢圖
        ax2 = self.axes[0, 1]
        n_recent = min(50, len(self.df))
        recent_sums = main_matrix[:n_recent].sum(axis=1)[::-1]
        x_range = range(1, n_recent + 1)
        ax2.plot(x_range, recent_sums, marker="o", color="#2980b9", linewidth=1.6, markersize=3.5, label="單期6碼總和")
        ax2.axhline(150, color="#27ae60", linestyle="-", linewidth=1.1, label="理論均值 (150)")
        ax2.fill_between(x_range, 115, 185, color="#2ecc71", alpha=0.15, label="黃金濾網區間 (115~185)")
        ax2.set_title(f"近 {n_recent} 期 6 碼總和走勢圖 (Sum Trend)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("期數時間軸 (由舊至新)", fontsize=8)
        ax2.set_ylabel("6碼加總", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(linestyle="--", alpha=0.4)

        # 圖 3：1~49號 當前遺漏期數走勢
        ax3 = self.axes[1, 0]
        gaps = res["gaps"]
        gap_colors = ["#e67e22" if 4 <= g <= 10 else "#95a5a6" for g in gaps]
        ax3.bar(range(1, 50), gaps, color=gap_colors)
        ax3.axhspan(4, 10, color="#f39c12", alpha=0.15, label="黃金回補期 (4~10期未開)")
        ax3.set_title("1~49號 當前遺漏期數 (橘色：處於黃金回補區間)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("獎號 (1~49)", fontsize=8)
        ax3.set_ylabel("連續未開期數", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        # 圖 4：特別號加權動能 Top 12
        ax4 = self.axes[1, 1]
        spec_series = pd.Series(res["spec_scores"]).sort_values(ascending=False).head(12)
        ax4.barh([f"{int(k):02d}號" for k in spec_series.index[::-1]], spec_series.values[::-1], color="#8e44ad")
        ax4.set_title("特別號加權動能 Top 12 (衝刺六獎 3+1 關鍵)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("加權積分 (歷史0.6 + 近百期2.5)", fontsize=8)
        ax4.grid(axis="x", linestyle="--", alpha=0.4)

        self.fig.tight_layout()
        self.canvas.draw()

    def export_predictions(self):
        """將當前推薦結果匯出為文字檔"""
        if not self.latest_result:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"大樂透推薦組合_{datetime.now().strftime('%Y%m%d')}.txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not save_path:
            return

        res = self.latest_result
        lines = [
            "==========================================================",
            f" 台灣大樂透 智慧縮水包牌推薦報告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            f" 歷史資料期數：共 {len(self.df)} 期 (最新一期：{self.df.iloc[0]['開獎日期']})",
            "==========================================================",
            f"【精選主候選池】：{', '.join(f'{n:02d}' for n in res['candidate_pool'])}",
            f"【特別號 Top 3】：{', '.join(f'{n:02d}' for n in res['top_specials'])}",
            f"【3碼組合覆蓋率】：{res['coverage_pct']:.2f}%",
            "----------------------------------------------------------",
        ]
        for idx, t in enumerate(res["tickets"], 1):
            t_str = ", ".join(f"{n:02d}" for n in t)
            odds = sum(1 for n in t if n % 2 == 1)
            highs = sum(1 for n in t if n >= 25)
            lines.append(f"第 {idx:02d} 注：[{t_str}] | 總和:{sum(t):3d} | 奇偶:{odds}:{6-odds} | 大小:{highs}:{6-highs}")
        lines.append("==========================================================")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("匯出成功", f"推薦組合已成功儲存至：\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BigLottoGUI(root, csv_path="taiwan_biglotto_2004_2026.csv")
    root.mainloop()
