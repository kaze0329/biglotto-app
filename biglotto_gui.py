import os
import platform
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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


class BigLottoMLGUI:
    def __init__(self, root: tk.Tk, csv_path: str = "taiwan_biglotto_2004_2026.csv"):
        self.root = root
        self.csv_path = csv_path
        self.root.title("台灣大樂透 機器學習時序預測系統 (GBDT 梯度提升樹 + 馬可夫狀態轉移模型)")
        self.root.geometry("1480x890")
        self.root.minsize(1280, 780)
        self.root.configure(bg="#f0f2f5")

        self.num_cols = ["獎號1", "獎號2", "獎號3", "獎號4", "獎號5", "獎號6"]
        self.all7_cols = self.num_cols + ["特別號"]
        self.df = self._load_or_init_data()
        self.latest_result = {}

        self._setup_styles()
        self._build_ui()
        self.run_analysis_and_update()

    def _load_or_init_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            for col in self.all7_cols:
                df[col] = df[col].astype(int)
            return df
        else:
            cols = ["開獎日期"] + self.all7_cols
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
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_paned, width=560)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=2)
        main_paned.add(right_panel, weight=5)

        # ==================== 左側面板 ====================
        # 1. 手動輸入歷史獎號區
        input_frame = ttk.LabelFrame(left_panel, text=" 📌 手動新增最新開獎紀錄 (即時重訓機器學習模型) ", padding=10)
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

        ttk.Label(row2, text=" 開獎特別號:", foreground="#c0392b", font=("Microsoft JhengHei", 9, "bold")).pack(side=tk.LEFT, padx=(6, 2))
        self.spec_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.spec_var, width=4, justify="center", font=("Consolas", 11, "bold"), foreground="#c0392b").pack(side=tk.LEFT, padx=2)

        btn_add = ttk.Button(input_frame, text="＋ 新增開獎獎號並重新訓練模型", style="Success.TButton", command=self.add_new_draw)
        btn_add.pack(fill=tk.X, pady=(4, 0))

        # 2. 模型超參數與縮水設定區
        param_frame = ttk.LabelFrame(left_panel, text=" ⚙️ 機器學習模型與縮水參數設定 ", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 8))

        p_row = ttk.Frame(param_frame)
        p_row.pack(fill=tk.X, pady=3)
        ttk.Label(p_row, text="預測候選池 (碼):").pack(side=tk.LEFT)
        self.pool_size_var = tk.IntVar(value=12)
        ttk.Spinbox(p_row, from_=9, to=18, textvariable=self.pool_size_var, width=4, font=("Consolas", 10)).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(p_row, text="推薦注數:").pack(side=tk.LEFT)
        self.ticket_count_var = tk.IntVar(value=6)
        ttk.Spinbox(p_row, from_=3, to=15, textvariable=self.ticket_count_var, width=4, font=("Consolas", 10)).pack(side=tk.LEFT, padx=(3, 10))

        ttk.Label(p_row, text="訓練視窗 (期):").pack(side=tk.LEFT)
        self.train_win_var = tk.IntVar(value=800)
        ttk.Spinbox(p_row, from_=200, to=2400, increment=100, textvariable=self.train_win_var, width=5, font=("Consolas", 10)).pack(side=tk.LEFT, padx=3)

        b_row = ttk.Frame(param_frame)
        b_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(b_row, text="⚡ 重新訓練模型與推論", style="Primary.TButton", command=self.run_analysis_and_update).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(b_row, text="💾 匯出模型預測報告 (.txt)", command=self.export_predictions).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # 3. 模型推論結果顯示區
        res_frame = ttk.LabelFrame(left_panel, text=" 🧠 GBDT + 馬可夫模型 下一期推論 6 碼組合 (無獨立特別號) ", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.info_label = ttk.Label(res_frame, text="模型訓練與推論中...", font=("Microsoft JhengHei", 9, "bold"), foreground="#16a085", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X, pady=(0, 6))

        cols = ("注別", "模型推論 6 碼組合", "期望機率均值", "6碼總和", "單雙比", "大小比")
        self.ticket_tree = ttk.Treeview(res_frame, columns=cols, show="headings", height=7)
        col_widths = [48, 185, 85, 55, 55, 55]
        for c, w in zip(cols, col_widths):
            self.ticket_tree.heading(c, text=c)
            self.ticket_tree.column(c, width=w, anchor="center")
        self.ticket_tree.pack(fill=tk.BOTH, expand=True)

        # 4. 最新歷史開獎紀錄檢視區
        hist_frame = ttk.LabelFrame(left_panel, text=" 📜 近期歷史開獎紀錄 (最新 15 期) ", padding=10)
        hist_frame.pack(fill=tk.BOTH, expand=True)

        h_cols = ("開獎日期", "開獎獎號 (6碼)", "特別號", "6碼總和")
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
        chart_frame = ttk.LabelFrame(right_panel, text=" 📊 機器學習條件機率、馬可夫轉移矩陣與滾動回測儀表板 ", padding=8)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(9.5, 7.2), facecolor="#f8f9fa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    @staticmethod
    def _extract_features_at_t(Y_hist: np.ndarray, markov_mat: np.ndarray) -> np.ndarray:
        """
        根據截至第 t 期的歷史二元矩陣 Y_hist (shape: [t+1, 49])，
        為 1~49 號各自提取 12 項時序狀態特徵，用於預測第 t+1 期開出機率。
        """
        curr_t = Y_hist.shape[0] - 1
        feats = np.zeros((49, 12), dtype=np.float32)

        # 特徵 0~2: 前 1, 2, 3 期狀態 (Lag-1, Lag-2, Lag-3)
        feats[:, 0] = Y_hist[curr_t]
        feats[:, 1] = Y_hist[curr_t - 1] if curr_t >= 1 else 0.0
        feats[:, 2] = Y_hist[curr_t - 2] if curr_t >= 2 else 0.0

        # 特徵 3~6: 多尺度移動平均動能 (MA5, MA10, MA20, MA50)
        feats[:, 3] = Y_hist[max(0, curr_t - 4):curr_t + 1].mean(axis=0)
        feats[:, 4] = Y_hist[max(0, curr_t - 9):curr_t + 1].mean(axis=0)
        feats[:, 5] = Y_hist[max(0, curr_t - 19):curr_t + 1].mean(axis=0)
        feats[:, 6] = Y_hist[max(0, curr_t - 49):curr_t + 1].mean(axis=0)

        # 特徵 7~9: 當前遺漏期數 (Gap1)、前次休眠間隔 (Gap2)、韋伯生存風險比 (Hazard Ratio)
        for n in range(49):
            ones = np.where(Y_hist[:, n] == 1.0)[0]
            if len(ones) == 0:
                gap1, gap2 = curr_t + 1, curr_t + 1
            elif len(ones) == 1:
                gap1, gap2 = curr_t - ones[-1], ones[-1] + 1
            else:
                gap1, gap2 = curr_t - ones[-1], ones[-1] - ones[-2] - 1
            feats[n, 7] = gap1
            feats[n, 8] = gap2
            mean_gap = (curr_t + 1) / (len(ones) + 1.0)
            feats[n, 9] = gap1 / (mean_gap + 1e-5)

        # 特徵 10~11: 一階與二階馬可夫條件轉移機率 (Markov-1 & Markov-2)
        active_curr = np.where(Y_hist[curr_t] == 1.0)[0]
        if len(active_curr) > 0:
            feats[:, 10] = markov_mat[active_curr, :].mean(axis=0)

        if curr_t >= 1:
            active_prev = np.where(Y_hist[curr_t - 1] == 1.0)[0]
            if len(active_prev) > 0:
                feats[:, 11] = markov_mat[active_prev, :].mean(axis=0)

        return feats

    def _train_and_predict(self, pool_size: int, num_tickets: int, train_window: int) -> dict:
        """建立時序資料集、訓練 GBDT 梯度提升樹與馬可夫鏈模型，並推論下一期號碼"""
        # 將資料依時間由舊至新排列 (index 0 為 2004 年，index T-1 為最新一期)
        df_chron = self.df.iloc[::-1].reset_index(drop=True)
        T = len(df_chron)
        all7_vals = df_chron[self.all7_cols].values

        Y = np.zeros((T, 49), dtype=np.float32)
        for t in range(T):
            for num in all7_vals[t]:
                Y[t, int(num) - 1] = 1.0

        # 1. 建立一階馬可夫狀態轉移矩陣 P(N_t+1 = j | N_t = i) (含拉普拉斯平滑)
        trans_counts = np.dot(Y[:-1].T, Y[1:]) + 1.0
        row_sums = Y[:-1].sum(axis=0, keepdims=True).T + 49.0
        markov_prob = trans_counts / row_sums

        # 同期共現親和矩陣 P(j | i in same draw)
        co_counts = np.dot(Y.T, Y)
        np.fill_diagonal(co_counts, 0)
        co_prob = co_counts / (Y.sum(axis=0, keepdims=True).T + 1e-6)

        # 2. 建構監督式機器學習訓練樣本 (展開時序視窗)
        start_t = max(55, T - train_window)
        X_list, y_list = [], []
        for t in range(start_t, T - 1):
            X_list.append(self._extract_features_at_t(Y[:t + 1], markov_prob))
            y_list.append(Y[t + 1])

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # 3. 訓練 HistGradientBoostingClassifier 模型
        clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=30,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # 4. 推論下一期 (T) 的各號碼開出機率 P(Y_T = 1 | X_{T-1})
        X_next = self._extract_features_at_t(Y, markov_prob)
        ml_probs = clf.predict_proba(X_next)[:, 1]

        latest_active = np.where(Y[-1] == 1.0)[0]
        markov_next_prob = markov_prob[latest_active, :].mean(axis=0)

        # 融合 GBDT 非線性機率 (70%) 與馬可夫轉移機率 (30%)
        ensemble_prob = 0.70 * ml_probs + 0.30 * markov_next_prob

        ranked_indices = np.argsort(ensemble_prob)[::-1]
        candidate_pool = sorted([int(i + 1) for i in ranked_indices[:pool_size]])

        # 5. 計算近 30 期 Walk-Forward 滾動回測命中表現
        hits_history = []
        bt_start = max(60, T - 30)
        for t in range(bt_start, T):
            X_bt = self._extract_features_at_t(Y[:t], markov_prob)
            p_ml_bt = clf.predict_proba(X_bt)[:, 1]
            act_prev = np.where(Y[t - 1] == 1.0)[0]
            p_mk_bt = markov_prob[act_prev, :].mean(axis=0)
            p_ens_bt = 0.70 * p_ml_bt + 0.30 * p_mk_bt
            top_k_bt = np.argsort(p_ens_bt)[::-1][:pool_size]
            actual_bt = np.where(Y[t] == 1.0)[0]
            hits_history.append(len(set(top_k_bt).intersection(set(actual_bt))))

        # 6. 結構濾網與貪婪集合覆蓋縮水
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

        uncovered_triplets = set(combinations(candidate_pool, 3))
        total_triplets_count = max(len(uncovered_triplets), 1)
        selected_tickets = []
        remaining = valid_combos.copy()

        for _ in range(min(num_tickets, len(remaining))):
            best_c, best_g, best_syn = None, -1, -1.0
            for c in remaining:
                g = len(set(combinations(c, 3)).intersection(uncovered_triplets))
                syn = sum(ensemble_prob[n - 1] for n in c) + sum(co_prob[p[0] - 1, p[1] - 1] * 0.2 for p in combinations(c, 2))
                if g > best_g or (g == best_g and syn > best_syn):
                    best_g, best_syn, best_c = g, syn, c
            selected_tickets.append(best_c)
            uncovered_triplets -= set(combinations(best_c, 3))
            remaining.remove(best_c)

        coverage_pct = (1.0 - len(uncovered_triplets) / total_triplets_count) * 100

        return {
            "ensemble_prob": ensemble_prob,
            "ml_probs": ml_probs,
            "markov_next_prob": markov_next_prob,
            "X_next": X_next,
            "candidate_pool": candidate_pool,
            "ranked_top_nums": [int(i + 1) for i in ranked_indices[:pool_size]],
            "tickets": selected_tickets,
            "coverage_pct": coverage_pct,
            "hits_history": hits_history
        }

    def add_new_draw(self):
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showerror("輸入錯誤", "請輸入開獎日期 (格式如 2026/11/03)！")
            return

        try:
            nums = [int(v.get().strip()) for v in self.num_vars]
            spec = int(self.spec_var.get().strip())
        except ValueError:
            messagebox.showerror("輸入錯誤", "請確保 6 個一般號與開獎特別號皆為整數數字！")
            return

        all_seven = nums + [spec]
        if any(n < 1 or n > 49 for n in all_seven):
            messagebox.showerror("範圍錯誤", "所有獎號必須介於 1 到 49 之間！")
            return

        if len(set(all_seven)) != 7:
            messagebox.showerror("重複錯誤", "同期的 6 個一般號與特別號之間不可有重複號碼！")
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

        self.df = pd.concat([new_row, self.df], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")

        for v in self.num_vars:
            v.set("")
        self.spec_var.set("")

        self.run_analysis_and_update()
        messagebox.showinfo("模型重訓完成", f"已成功寫入 {date_str} 開獎紀錄，並重新訓練 GBDT 與馬可夫預測模型！")

    def load_custom_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            self.df = self._load_or_init_data()
            self.run_analysis_and_update()

    def run_analysis_and_update(self):
        if len(self.df) < 60:
            messagebox.showwarning("資料不足", "歷史期數需至少 60 期以上方可建立機器學習時序特徵！")
            return

        pool_size = self.pool_size_var.get()
        num_tickets = self.ticket_count_var.get()
        train_win = self.train_win_var.get()

        res = self._train_and_predict(pool_size, num_tickets, train_win)
        self.latest_result = res

        # 依預測機率高低列出前 8 名與完整候選池
        top_prob_str = ", ".join(
            f"{n:02d}({res['ensemble_prob'][n-1]*100:.1f}%)" for n in res["ranked_top_nums"][:6]
        )
        pool_str = ", ".join(f"{n:02d}" for n in res["candidate_pool"])
        latest_date = self.df.iloc[0]["開獎日期"]

        self.info_label.config(
            text=(
                f"🧠 模型訓練視窗：近 {train_win} 期 (最新：{latest_date}) | 近30期平均命中：{np.mean(res['hits_history']):.2f} 碼\n"
                f"📈 條件機率最高前6碼：{top_prob_str}\n"
                f"🎯 模型推論 {pool_size} 碼候選池：[{pool_str}] (3碼覆蓋率 {res['coverage_pct']:.1f}%)"
            )
        )

        # 更新推薦注數表格
        for item in self.ticket_tree.get_children():
            self.ticket_tree.delete(item)

        for idx, t in enumerate(res["tickets"], 1):
            t_str = "  ".join(f"{n:02d}" for n in t)
            avg_p = np.mean([res["ensemble_prob"][n - 1] * 100 for n in t])
            s_val = sum(t)
            odds = sum(1 for n in t if n % 2 == 1)
            highs = sum(1 for n in t if n >= 25)
            self.ticket_tree.insert(
                "", tk.END,
                values=(f"第 {idx:02d} 注", t_str, f"{avg_p:.2f}%", s_val, f"{odds}:{6-odds}", f"{highs}:{6-highs}")
            )

        # 更新歷史紀錄表格
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)

        for _, row in self.df.head(15).iterrows():
            nums = [int(row[c]) for c in self.num_cols]
            n_str = " ".join(f"{n:02d}" for n in nums)
            self.hist_tree.insert("", tk.END, values=(row["開獎日期"], n_str, f"{int(row['特別號']):02d}", sum(nums)))

        self._update_charts(res)

    def _update_charts(self, res: dict):
        for ax in self.axes.flat:
            ax.clear()

        cand_set = set(res["candidate_pool"])
        ens_prob = res["ensemble_prob"] * 100
        mk_prob = res["markov_next_prob"] * 100

        # 圖 1：1~49號 機器學習預測下期開出條件機率
        ax1 = self.axes[0, 0]
        colors1 = ["#e74c3c" if (i + 1) in cand_set else "#95a5a6" for i in range(49)]
        ax1.bar(range(1, 50), ens_prob, color=colors1)
        ax1.axhline((7 / 49) * 100, color="#2c3e50", linestyle="--", linewidth=1.1, label="隨機基準機率 (14.29%)")
        ax1.set_title(f"1~49號 下期開出條件機率 P(Y_t+1|X_t) (紅：Top {len(cand_set)} 候選池)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("獎號 (1~49)", fontsize=8)
        ax1.set_ylabel("模型預測機率 (%)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # 圖 2：由最新一期獎號觸發之一階馬可夫轉移機率 Top 12
        ax2 = self.axes[0, 1]
        top_mk_idx = np.argsort(mk_prob)[::-1][:12]
        mk_labels = [f"{i + 1:02d}號" for i in top_mk_idx][::-1]
        mk_vals = [mk_prob[i] for i in top_mk_idx][::-1]
        ax2.barh(mk_labels, mk_vals, color="#2980b9")
        ax2.axvline((7 / 49) * 100, color="#e74c3c", linestyle="--", linewidth=1.1, label="基準轉移率 (14.29%)")
        ax2.set_title("一階馬可夫鏈狀態轉移機率 Top 12 (由上期開出獎號觸發)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("跨期條件轉移機率 (%)", fontsize=8)
        ax2.legend(loc="lower right", fontsize=7)
        ax2.grid(axis="x", linestyle="--", alpha=0.4)

        # 圖 3：韋伯生存風險比 (Weibull Hazard Ratio: 當前遺漏 / 歷史平均間隔)
        ax3 = self.axes[1, 0]
        hazard_ratios = res["X_next"][:, 9]
        hz_colors = ["#8e44ad" if (i + 1) in cand_set else "#bdc3c7" for i in range(49)]
        ax3.bar(range(1, 50), hazard_ratios, color=hz_colors)
        ax3.axhline(1.0, color="#d35400", linestyle="--", linewidth=1.1, label="平均休眠週期臨界線 (Hazard=1.0)")
        ax3.set_title("1~49號 韋伯生存風險比 (Hazard Ratio，紫色：入選模型候選池)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("獎號 (1~49)", fontsize=8)
        ax3.set_ylabel("當前遺漏 / 個人歷史均值", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        # 圖 4：近 30 期 Walk-Forward 滾動回測命中數走勢
        ax4 = self.axes[1, 1]
        hits = res["hits_history"]
        x_bt = range(1, len(hits) + 1)
        expected_random = len(cand_set) * 7.0 / 49.0
        ax4.plot(x_bt, hits, marker="s", color="#16a085", linewidth=1.6, markersize=3.5, label=f"模型實際命中 (均值:{np.mean(hits):.2f}碼)")
        ax4.axhline(expected_random, color="#e67e22", linestyle="--", linewidth=1.3, label=f"隨機盲選期望值 ({expected_random:.2f}碼)")
        ax4.axhline(3.0, color="#c0392b", linestyle=":", linewidth=1.1, label="3碼命中門檻線")
        ax4.set_title(f"近 {len(hits)} 期滾動回測：模型候選池實際命中獎號顆數", fontsize=10, fontweight="bold")
        ax4.set_xlabel("近期時間軸 (由舊至新)", fontsize=8)
        ax4.set_ylabel("單期命中顆數", fontsize=8)
        ax4.set_yticks(range(0, 8))
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(linestyle="--", alpha=0.4)

        self.fig.tight_layout()
        self.canvas.draw()

    def export_predictions(self):
        if not self.latest_result:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"大樂透ML模型推論報告_{datetime.now().strftime('%Y%m%d')}.txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not save_path:
            return

        res = self.latest_result
        lines = [
            "==================================================================",
            f" 台灣大樂透 機器學習時序模型 (GBDT + 馬可夫鏈) 推論報告",
            f" 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 最新資料期別：{self.df.iloc[0]['開獎日期']}",
            "==================================================================",
            "【模型預測條件機率排行】：",
        ]
        for n in res["ranked_top_nums"]:
            p_ens = res["ensemble_prob"][n - 1] * 100
            p_ml = res["ml_probs"][n - 1] * 100
            p_mk = res["markov_next_prob"][n - 1] * 100
            lines.append(f"  - {n:02d}號：綜合預測機率 {p_ens:.2f}% (GBDT={p_ml:.2f}%, 馬可夫轉移={p_mk:.2f}%)")

        lines.extend([
            "------------------------------------------------------------------",
            f"【候選池 3 碼縮水覆蓋率】：{res['coverage_pct']:.2f}%",
            "【模型推論最佳化 6 碼包牌組合】：",
        ])
        for idx, t in enumerate(res["tickets"], 1):
            t_str = ", ".join(f"{n:02d}" for n in t)
            avg_p = np.mean([res["ensemble_prob"][n - 1] * 100 for n in t])
            odds = sum(1 for n in t if n % 2 == 1)
            highs = sum(1 for n in t if n >= 25)
            lines.append(f"  第 {idx:02d} 注：[{t_str}] | 期望機率均值:{avg_p:.2f}% | 總和:{sum(t):3d} | 奇偶:{odds}:{6-odds} | 大小:{highs}:{6-highs}")
        lines.append("==================================================================")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("匯出成功", f"機器學習推論報告已儲存至：\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BigLottoMLGUI(root, csv_path="taiwan_biglotto_2004_2026.csv")
    root.mainloop()
