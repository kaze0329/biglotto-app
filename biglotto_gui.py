import os
import platform
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
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


class PureNumpyMLClassifier:
    """
    純 NumPy 實作之機器學習二元分類器 (L2 正規化 Logistic Regression + 二階交互特徵擴展)
    完全不依賴外部 scikit-learn，徹底根除 PyInstaller 打包 ModuleNotFoundError 問題。
    """
    def __init__(self, lr: float = 0.08, epochs: int = 250, l2_reg: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def _expand_and_normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        # 加入非線性特徵轉換 (平方項與關鍵交互項：遺漏風險 x 馬可夫轉移率)
        hazard_x_mk1 = (X[:, 9] * X[:, 10]).reshape(-1, 1)
        ma5_x_mk1 = (X[:, 3] * X[:, 10]).reshape(-1, 1)
        X_ext = np.hstack([X, np.square(X[:, 3:12]), hazard_x_mk1, ma5_x_mk1])

        if fit:
            self.mean = X_ext.mean(axis=0, keepdims=True)
            self.std = X_ext.std(axis=0, keepdims=True) + 1e-6
        return (X_ext - self.mean) / self.std

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_norm = self._expand_and_normalize(X, fit=True)
        n_samples, n_features = X_norm.shape
        self.weights = np.zeros(n_features, dtype=np.float32)

        # 將初始偏置設為大樂透自然基準對數勝算 ln(p / (1-p)), p = 7/49
        p_base = 7.0 / 49.0
        self.bias = float(np.log(p_base / (1.0 - p_base)))

        for _ in range(self.epochs):
            linear = np.dot(X_norm, self.weights) + self.bias
            preds = self._sigmoid(linear)
            error = preds - y
            grad_w = (np.dot(X_norm.T, error) / n_samples) + self.l2_reg * self.weights
            grad_b = float(np.mean(error))
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._expand_and_normalize(X, fit=False)
        linear = np.dot(X_norm, self.weights) + self.bias
        return self._sigmoid(linear)


class BigLottoMLGUI:
    def __init__(self, root: tk.Tk, csv_path: str = "taiwan_biglotto_2004_2026.csv"):
        self.root = root
        self.csv_path = csv_path
        self.root.title("台灣大樂透 機器學習時序預測系統 (內建原生 ML 引擎 + 馬可夫狀態轉移模型)")
        self.root.geometry("1480x890")
        self.root.minsize(1280, 780)
        self.root.configure(bg="#f0f2f5")

        self.num_cols = ["獎號1", "獎號2", "獎號3", "獎號4", "獎號5", "獎號6"]
        self.all7_cols = self.num_cols + ["特別號"]
        self.df = self._load_or_init_data()
        self.latest_result = {}

        self._setup_styles()
        self._build_ui()
        if len(self.df) >= 60:
            self.run_analysis_and_update()

    def _load_or_init_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            for col in self.all7_cols:
                df[col] = df[col].astype(int)
            return df
        else:
            messagebox.showwarning(
                "找不到歷史檔案",
                f"在當前目錄找不到 {self.csv_path}！\n請點擊介面右上角「📂 載入其他 CSV」選擇你的歷史獎號檔。"
            )
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

        res_frame = ttk.LabelFrame(left_panel, text=" 🧠 機器學習 + 馬可夫模型 下一期推論 6 碼組合 (無獨立特別號) ", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.info_label = ttk.Label(res_frame, text="請載入 CSV 或點擊重新訓練...", font=("Microsoft JhengHei", 9, "bold"), foreground="#16a085", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X, pady=(0, 6))

        cols = ("注別", "模型推論 6 碼組合", "期望機率均值", "6碼總和", "單雙比", "大小比")
        self.ticket_tree = ttk.Treeview(res_frame, columns=cols, show="headings", height=7)
        col_widths = [48, 185, 85, 55, 55, 55]
        for c, w in zip(cols, col_widths):
            self.ticket_tree.heading(c, text=c)
            self.ticket_tree.column(c, width=w, anchor="center")
        self.ticket_tree.pack(fill=tk.BOTH, expand=True)

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
        curr_t = Y_hist.shape[0] - 1
        feats = np.zeros((49, 12), dtype=np.float32)

        feats[:, 0] = Y_hist[curr_t]
        feats[:, 1] = Y_hist[curr_t - 1] if curr_t >= 1 else 0.0
        feats[:, 2] = Y_hist[curr_t - 2] if curr_t >= 2 else 0.0

        feats[:, 3] = Y_hist[max(0, curr_t - 4):curr_t + 1].mean(axis=0)
        feats[:, 4] = Y_hist[max(0, curr_t - 9):curr_t + 1].mean(axis=0)
        feats[:, 5] = Y_hist[max(0, curr_t - 19):curr_t + 1].mean(axis=0)
        feats[:, 6] = Y_hist[max(0, curr_t - 49):curr_t + 1].mean(axis=0)

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

        active_curr = np.where(Y_hist[curr_t] == 1.0)[0]
        if len(active_curr) > 0:
            feats[:, 10] = markov_mat[active_curr, :].mean(axis=0)

        if curr_t >= 1:
            active_prev = np.where(Y_hist[curr_t - 1] == 1.0)[0]
            if len(active_prev) > 0:
                feats[:, 11] = markov_mat[active_prev, :].mean(axis=0)

        return feats

    def _train_and_predict(self, pool_size: int, num_tickets: int, train_window: int) -> dict:
        df_chron = self.df.iloc[::-1].reset_index(drop=True)
        T = len(df_chron)
        all7_vals = df_chron[self.all7_cols].values

        Y = np.zeros((T, 49), dtype=np.float32)
        for t in range(T):
            for num in all7_vals[t]:
                Y[t, int(num) - 1] = 1.0

        trans_counts = np.dot(Y[:-1].T, Y[1:]) + 1.0
        row_sums = Y[:-1].sum(axis=0, keepdims=True).T + 49.0
        markov_prob = trans_counts / row_sums

        co_counts = np.dot(Y.T, Y)
        np.fill_diagonal(co_counts, 0)
        co_prob = co_counts / (Y.sum(axis=0, keepdims=True).T + 1e-6)

        start_t = max(55, T - train_window)
        X_list, y_list = [], []
        for t in range(start_t, T - 1):
            X_list.append(self._extract_features_at_t(Y[:t + 1], markov_prob))
            y_list.append(Y[t + 1])

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        clf = PureNumpyMLClassifier(lr=0.08, epochs=250, l2_reg=0.01)
        clf.fit(X_train, y_train)

        X_next = self._extract_features_at_t(Y, markov_prob)
        ml_probs = clf.predict_proba(X_next)

        latest_active = np.where(Y[-1] == 1.0)[0]
        markov_next_prob = markov_prob[latest_active, :].mean(axis=0)

        ensemble_prob = 0.70 * ml_probs + 0.30 * markov_next_prob

        ranked_indices = np.argsort(ensemble_prob)[::-1]
        candidate_pool = sorted([int(i + 1) for i in ranked_indices[:pool_size]])

        hits_history = []
        bt_start = max(60, T - 30)
        for t in range(bt_start, T):
            X_bt = self._extract_features_at_t(Y[:t], markov_prob)
            p_ml_bt = clf.predict_proba(X_bt)
            act_prev = np.where(Y[t - 1] == 1.0)[0]
            p_mk_bt = markov_prob[act_prev, :].mean(axis=0)
            p_ens_bt = 0.70 * p_ml_bt + 0.30 * p_mk_bt
            top_k_bt = np.argsort(p_ens_bt)[::-1][:pool_size]
            actual_bt = np.where(Y[t] == 1.0)[0]
            hits_history.append(len(set(top_k_bt).intersection(set(actual_bt))))

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
        messagebox.showinfo("模型重訓完成", f"已成功寫入 {date_str} 開獎紀錄，並重新訓練時序預測模型！")

    def load_custom_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            self.df = self._load_or_init_data()
            if len(self.df) >= 60:
                self.run_analysis_and_update()

    def run_analysis_and_update(self):
        if len(self.df) < 60:
            messagebox.showwarning("資料不足", "請確認已載入歷史獎號 CSV 檔（需至少 60 期以上）！")
            return

        pool_size = self.pool_size_var.get()
        num_tickets = self.ticket_count_var.get()
        train_win = self.train_win_var.get()

        res = self._train_and_predict(pool_size, num_tickets, train_win)
        self.latest_result = res

        top_prob_str = ", ".join(
            f"{n:02d}({res['ensemble_prob'][n-1]*100:.1f}%)" for n in res["ranked_top_nums"][:6]
        )
        pool_str = ", ".join(f"{n:02d}" for n in res["candidate_pool"])
        latest_date = self.df.iloc[0]["開獎日期"]

        self.info_label.config(
