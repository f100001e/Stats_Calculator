import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import zscore, norm


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, Toplevel, Scrollbar, Canvas, Frame, Text, Label, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import gamma, binom, norm as stats, gaussian_kde


# ─── Utility: Style Buttons ───────────────────────────────────────────────────
def style_button(btn):
    btn.config(
        bg="white", fg="black", font=("Arial", 12, "bold"),
        activebackground="#cccccc", activeforeground="black",
        borderwidth=2, highlightthickness=0, relief="solid",
        highlightbackground="white", highlightcolor="white",
        disabledforeground="black", takefocus=False, padx=10, pady=5
    )


# ─── Helper: Save Figure ──────────────────────────────────────────────────────
def _save_plot(fig):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("Saved", f"Plot saved as {file_path}")


# ─── 1) Dataset Visualization (Scatter, Line, Box, Histogram + Stats) ───────
def dataset_visualization(root):
    window = Toplevel(root)
    window.title("Dataset Visualization")
    window.geometry("1250x900")
    window.configure(bg="#333")

    # Scrollable Canvas
    canvas = Canvas(window, bg="#333")
    scroll_y = Scrollbar(window, orient="vertical", command=canvas.yview)
    scroll_x = Scrollbar(window, orient="horizontal", command=canvas.xview)

    frame = Frame(canvas, bg="#333")
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    # Prompt for X and Y values
    x_input = simpledialog.askstring("Input", "Enter X values (comma-separated):", parent=window)
    y_input = simpledialog.askstring("Input", "Enter Y values (comma-separated):", parent=window)

    try:
        x_values = list(map(float, x_input.split(",")))
        y_values = list(map(float, y_input.split(",")))
        if len(x_values) != len(y_values):
            messagebox.showerror("Error", "X and Y values must have the same length!")
            return
        df = pd.DataFrame({'x': x_values, 'y': y_values})

    except Exception:
        messagebox.showerror("Error", "Invalid numeric input! Please enter numbers separated by commas.")
        return

    visualize_data(df, frame, canvas)


def visualize_data(df, frame, canvas):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.set_style("darkgrid")

    # Top-left: Scatter x vs y
    sns.scatterplot(x=df['x'], y=df['y'], ax=axes[0, 0])
    axes[0, 0].set_title("Scatter: x vs y")

    # Top-right: Line x vs y
    sns.lineplot(x=df['x'], y=df['y'], ax=axes[0, 1], marker='o')
    axes[0, 1].set_title("Line: x vs y")

    # Bottom-left: Boxplot of y
    sns.boxplot(y=df['y'], ax=axes[1, 0], showmeans=True, meanline=True)
    axes[1, 0].set_title("Boxplot of y")

    # Bottom-right: Histogram + KDE of y
    sns.histplot(df['y'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Histogram + KDE of y")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("P(y)")

    plt.tight_layout()

    # Embed in Tkinter
    canvas_plot = FigureCanvasTkAgg(fig, master=frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack()

    # Descriptive statistics in a scrollable Text widget
    stats_info = df.describe().to_string()
    text_frame = Frame(frame, bg="#333")
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)

    text_widget = Text(text_frame, height=12, wrap="none", bg="#333", fg="white", font=("Arial", 12))
    text_widget.insert("1.0", stats_info)
    text_widget.config(state="disabled")

    scroll_y = Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    text_widget.pack(fill="both", expand=True)


# ─── 2) PDF Visualization ─────────────────────────────────────────────────────
def pdf_visualization(root):
    window = Toplevel(root)
    window.title("PDF Visualization")
    window.geometry("900x700")
    window.configure(bg="#333")

    # Scrollable Canvas
    canvas = Canvas(window, bg="#333")
    scroll_y = Scrollbar(window, orient="vertical", command=canvas.yview)
    scroll_x = Scrollbar(window, orient="horizontal", command=canvas.xview)

    frame = Frame(canvas, bg="#333")
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    # Prompt for dataset
    x_input = simpledialog.askstring("Input", "Enter X values (comma-separated):", parent=window)
    y_input = simpledialog.askstring("Input", "Enter Y values (comma-separated):", parent=window)
    try:
        x_values = list(map(float, x_input.split(",")))
        y_values = list(map(float, y_input.split(",")))
        if len(x_values) != len(y_values):
            messagebox.showerror("Error", "X and Y values must have the same length!")
            return
        df = pd.DataFrame({'x': x_values, 'y': y_values})
    except Exception:
        messagebox.showerror("Error", "Invalid numeric input! Please enter numbers separated by commas.")
        return

    # Ask which column to plot PDF for
    col_choice = simpledialog.askstring("Column", "Plot PDF of 'x' or 'y'?", parent=window)
    if col_choice is None or col_choice.strip().lower() not in df.columns:
        messagebox.showerror("Error", "Invalid column selection. Choose 'x' or 'y'.")
        return
    col_choice = col_choice.strip().lower()

    data = df[col_choice]
    stats_info = data.describe().to_string()

    # Plot KDE-based PDF
    fig, ax = plt.subplots(figsize=(8, 5))
    kde_est = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = kde_est(x_vals)
    ax.plot(x_vals, y_vals, color='cyan', label=f"PDF of {col_choice}")
    ax.set_title(f"Kernel-Density PDF of {col_choice}")
    ax.set_xlabel(col_choice)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Embed plot
    canvas_plot = FigureCanvasTkAgg(fig, master=frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(pady=(10, 20))

    # Descriptive stats below
    text_frame = Frame(frame, bg="#333")
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    text_widget = Text(text_frame, height=12, wrap="none", bg="#333", fg="white", font=("Arial", 12))
    text_widget.insert("1.0", stats_info)
    text_widget.config(state="disabled")
    scroll_y2 = Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll_y2.set)
    scroll_y2.pack(side="right", fill="y")
    text_widget.pack(fill="both", expand=True)

    # Save button
    save_button = tk.Button(window, text="Save PDF", command=lambda: _save_plot(fig), font=("Arial", 12, "bold"))
    style_button(save_button)
    save_button.pack(pady=10)


# ─── 3) CDF Visualization ─────────────────────────────────────────────────────
def cdf_visualization(root):
    window = Toplevel(root)
    window.title("CDF Visualization")
    window.geometry("900x700")
    window.configure(bg="#333")

    # Scrollable Canvas
    canvas = Canvas(window, bg="#333")
    scroll_y = Scrollbar(window, orient="vertical", command=canvas.yview)
    scroll_x = Scrollbar(window, orient="horizontal", command=canvas.xview)

    frame = Frame(canvas, bg="#333")
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    # Prompt for dataset
    x_input = simpledialog.askstring("Input", "Enter X values (comma-separated):", parent=window)
    y_input = simpledialog.askstring("Input", "Enter Y values (comma-separated):", parent=window)

    try:
        x_values = list(map(float, x_input.split(",")))
        y_values = list(map(float, y_input.split(",")))
        if len(x_values) != len(y_values):
            messagebox.showerror("Error", "X and Y values must have the same length!")
            return
        df = pd.DataFrame({'x': x_values, 'y': y_values})
    except Exception:
        messagebox.showerror("Error", "Invalid numeric input! Please enter numbers separated by commas.")
        return

    # Ask which column to plot CDF for
    col_choice = simpledialog.askstring("Column", "Plot CDF of 'x' or 'y'?", parent=window)
    if col_choice is None or col_choice.strip().lower() not in df.columns:
        messagebox.showerror("Error", "Invalid column selection. Choose 'x' or 'y'.")
        return
    col_choice = col_choice.strip().lower()

    # Sort and process data
    data = np.sort(df[col_choice])
    ecdf = np.arange(1, len(data) + 1) / len(data)
    stats_info = df[col_choice].describe().to_string()

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(data, ecdf, where='post', color='orange', linewidth=2.5, label="Empirical CDF")


    # Prompt for value to evaluate
    x_eval = simpledialog.askstring(
        "Evaluate CDF",
        f"Enter a value for ECDF on '{col_choice}' (or 'min', 'max', 'mean'):",
        parent=window
    )

    try:
        if x_eval:
            x_eval = x_eval.strip().lower()
            if x_eval == 'max':
                x_val = data.max()
            elif x_eval == 'min':
                x_val = data.min()
            elif x_eval == 'mean':
                x_val = data.mean()
            else:
                x_val = float(x_eval)

            percentile = np.mean(data <= x_val)
            ax.axvline(x=x_val, color='red', linestyle='--',
                       label=f'Input = {x_val}\nPercentile ≈ {percentile:.2%}')
            ax.axhline(y=percentile, color='red', linestyle=':')
            ax.legend()
            messagebox.showinfo("ECDF Value", f"Empirical CDF at x = {x_val} is approximately {percentile:.4f}")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")
        return

    # Final plot styling
    ax.set_title(f"Empirical CDF of '{col_choice}'")
    ax.set_xlabel(col_choice)
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    plt.tight_layout()

    # Embed plot
    canvas_plot = FigureCanvasTkAgg(fig, master=frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(pady=(10, 20))

    # Descriptive stats
    text_frame = Frame(frame, bg="#333")
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    text_widget = Text(text_frame, height=12, wrap="none", bg="#333", fg="white", font=("Arial", 12))
    text_widget.insert("1.0", stats_info)
    text_widget.config(state="disabled")
    scroll_y2 = Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll_y2.set)
    scroll_y2.pack(side="right", fill="y")
    text_widget.pack(fill="both", expand=True)

    # Save button
    save_button = tk.Button(window, text="Save CDF", command=lambda: _save_plot(fig), font=("Arial", 12, "bold"))
    style_button(save_button)
    save_button.pack(pady=10)


# ─── 4) Binomial Distribution Calculator ──────────────────────────────────────
def plot_binomial_graph(n, p, x, selected_type, parent_frame):
    for widget in parent_frame.winfo_children():
        widget.destroy()

    x_vals = np.arange(0, n + 1)
    y_vals = binom.pmf(x_vals, n, p)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [
        "red" if
        (selected_type == "P(X = x)" and val == x) or
        (selected_type == "P(X ≤ x)" and val <= x) or
        (selected_type == "P(X ≥ x)" and val >= x)
        else "gray"
        for val in x_vals
    ]

    ax.bar(x_vals, y_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Number of Successes (k)")
    ax.set_ylabel("PMF")
    ax.set_title(f"Binomial(n={n}, p={p:.2f})")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


def binomial_distribution():
    window = Toplevel()
    window.title("Binomial Distribution")
    window.geometry("600x500")
    window.configure(bg="#333")

    n = simpledialog.askinteger("Input", "Enter number of trials (n):", parent=window)
    p = simpledialog.askfloat("Input", "Enter probability of success (p):", parent=window)
    x = simpledialog.askinteger("Input", "Enter value to calculate (x):", parent=window)

    if n is None or p is None or x is None or n <= 0 or p < 0 or p > 1 or x < 0 or x > n:
        messagebox.showerror(
            "Error",
            "Invalid parameters! n must be positive, p must be between 0 and 1, and x must be between 0 and n."
        )
        return

    plot_frame = tk.Frame(window, bg="#333")
    plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    prob_types = ["P(X = x)", "P(X ≤ x)", "P(X ≥ x)"]
    buttons_frame = tk.Frame(window, bg="#333")
    buttons_frame.pack(fill="x", padx=10, pady=5)

    def update_prob_display(selected_type):
        if selected_type == "P(X = x)":
            prob = binom.pmf(x, n, p)
        elif selected_type == "P(X ≤ x)":
            prob = binom.cdf(x, n, p)
        else:  # P(X ≥ x)
            prob = 1 - binom.cdf(x - 1, n, p)

        result_label.config(text=f"{selected_type} = {prob:.6f}")
        plot_binomial_graph(n, p, x, selected_type, plot_frame)

    for prob_type in prob_types:
        btn = tk.Button(
            buttons_frame,
            text=prob_type,
            command=lambda t=prob_type: update_prob_display(t)
        )
        style_button(btn)
        btn.pack(side="left", padx=5, expand=True)

    result_label = tk.Label(window, text="", font=("Arial", 14), bg="#333", fg="white")
    result_label.pack(pady=10)

    update_prob_display("P(X = x)")


# ─── 5) Gamma Distribution Calculator ─────────────────────────────────────────
def gamma_distribution():
    window = Toplevel()
    window.title("Gamma Distribution")
    window.geometry("600x500")
    window.configure(bg="#333")

    shape = simpledialog.askfloat("Input", "Enter shape parameter (k):", parent=window)
    scale = simpledialog.askfloat("Input", "Enter scale parameter (θ):", parent=window)

    if shape is None or scale is None or shape <= 0 or scale <= 0:
        messagebox.showerror("Error", "Shape and Scale must be positive numbers!")
        return

    x_vals = np.linspace(0, 20, 1000)
    y_vals = gamma.pdf(x_vals, a=shape, scale=scale)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, y_vals, label=f'Gamma PDF (k={shape}, θ={scale})', color='magenta')
    ax.set_title("Gamma Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    save_button = tk.Button(window, text="Save Plot", command=lambda: _save_plot(fig), font=("Arial", 12, "bold"))
    style_button(save_button)
    save_button.pack(pady=10)

    # Real-time coordinate display (optional)
    coordinates_label = tk.Label(window, text="X: 0.00, Y: 0.00", font=("Arial", 12), bg="#333", fg="white")
    coordinates_label.pack(pady=5)

    def update_coordinates(event):
        x_pix = event.x
        y_pix = event.y
        w = canvas.get_tk_widget().winfo_width()
        h = canvas.get_tk_widget().winfo_height()
        data_x = ax.get_xlim()[0] + (x_pix / w) * (ax.get_xlim()[1] - ax.get_xlim()[0])
        data_y = ax.get_ylim()[0] + ((h - y_pix) / h) * (ax.get_ylim()[1] - ax.get_ylim()[0])
        coordinates_label.config(text=f"X: {data_x:.2f}, Y: {data_y:.2f}")

    canvas.get_tk_widget().bind("<Motion>", update_coordinates)


# ─── 6) Confidence Interval Calculation ───────────────────────────────────────
def confidence_interval():
    window = Toplevel()
    window.title("Confidence Interval")
    window.geometry("600x500")
    window.configure(bg="#333")

    mean = simpledialog.askfloat("Input", "Enter the sample mean (x̄):", parent=window)
    std_dev = simpledialog.askfloat("Input", "Enter the sample standard deviation (s):", parent=window)
    n = simpledialog.askinteger("Input", "Enter the sample size (n):", parent=window)
    confidence_level = simpledialog.askfloat(
        "Input", "Enter the confidence level (e.g., 0.95 for 95%):", parent=window
    )

    if mean is None or std_dev is None or n is None or confidence_level is None:
        messagebox.showerror("Error", "All values must be entered!")
        return

    SE = std_dev / np.sqrt(n)
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence_level)
    if z is None:
        messagebox.showerror("Error", "Invalid confidence level! Use 0.90, 0.95, or 0.99.")
        return

    margin_of_error = z * SE
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(mean - 4 * SE, mean + 4 * SE, 1000)
    y = stats.pdf(x, loc=mean, scale=SE)

    ax.plot(x, y, 'b-', label='Sampling Distribution')
    ax.axvline(x=mean, color='black', linestyle='-', alpha=0.5, label='Sample Mean')
    ax.axvline(x=lower_bound, color='red', linestyle='--', label=f'{confidence_level * 100:.0f}% CI')
    ax.axvline(x=upper_bound, color='red', linestyle='--')
    ax.fill_between(x, 0, y, where=(x >= lower_bound) & (x <= upper_bound), color='blue', alpha=0.2)

    ax.set_title(f'{confidence_level * 100:.0f}% Confidence Interval')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    result_label = tk.Label(
        window,
        text=f"Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]",
        font=("Arial", 14), bg="#333", fg="white"
    )
    result_label.pack(pady=10)


# ─── 7) Hypothesis Testing Calculation ────────────────────────────────────────
def hypothesis_testing():
    window = Toplevel()
    window.title("Hypothesis Testing")
    window.geometry("700x600")
    window.configure(bg="#333")

    sample_mean = simpledialog.askfloat("Input", "Enter the sample mean (x̄):", parent=window)
    population_mean = simpledialog.askfloat("Input", "Enter the population mean (μ):", parent=window)
    std_dev = simpledialog.askfloat("Input", "Enter the population standard deviation (σ):", parent=window)
    n = simpledialog.askinteger("Input", "Enter the sample size (n):", parent=window)
    if sample_mean is None or population_mean is None or std_dev is None or n is None:
        messagebox.showerror("Error", "All values must be entered!")
        return

    SE = std_dev / np.sqrt(n)
    z = (sample_mean - population_mean) / SE

    test_type = StringVar(value="two-tailed")
    controls_frame = tk.Frame(window, bg="#333")
    controls_frame.pack(fill="x", padx=10, pady=10)
    graph_frame = tk.Frame(window, bg="#333")
    graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def toggle_test_type():
        if test_type.get() == "two-tailed":
            test_type.set("one-tailed")
            toggle_button.config(text="Switch to Two-Tailed Test")
        else:
            test_type.set("two-tailed")
            toggle_button.config(text="Switch to One-Tailed Test")
        update_graph()

    toggle_button = tk.Button(
        controls_frame,
        text="Switch to One-Tailed Test",
        command=toggle_test_type,
        font=("Arial", 12, "bold")
    )
    style_button(toggle_button)
    toggle_button.pack(side="left", padx=5)

    def calculate_p_value(z_val, t_type):
        if t_type == "two-tailed":
            return 2 * (1 - stats.cdf(abs(z_val)))
        else:
            return 1 - stats.cdf(z_val) if z_val > 0 else stats.cdf(z_val)

    initial_p = calculate_p_value(z, test_type.get())

    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    x_vals = np.linspace(-4, 4, 1000)
    y_vals = stats.pdf(x_vals)

    def update_graph():
        ax.clear()
        p_val = calculate_p_value(z, test_type.get())
        p_value_label.config(text=f"P-Value: {p_val:.4f}")

        ax.plot(x_vals, y_vals, label='Standard Normal Distribution', color='blue')
        if test_type.get() == "two-tailed":
            rejection_x = x_vals[(x_vals >= abs(z)) | (x_vals <= -abs(z))]
            rejection_y = stats.pdf(rejection_x)
            ax.fill_between(rejection_x, 0, rejection_y, color='red', alpha=0.3, label="Rejection Region (Two-Tailed)")
        else:
            if z > 0:
                rejection_x = x_vals[x_vals >= z]
            else:
                rejection_x = x_vals[x_vals <= z]
            rejection_y = stats.pdf(rejection_x)
            ax.fill_between(rejection_x, 0, rejection_y, color='red', alpha=0.3,
                            label=f"Rejection Region (One-Tailed, {'Right' if z>0 else 'Left'})")

        ax.axvline(x=z, color='black', linestyle='--', label=f'Z = {z:.4f}')
        ax.set_xlim(-4, max(4, z + 1))
        ax.set_ylim(0, stats.pdf(0) * 1.1)
        ax.legend()
        ax.set_title(f'Hypothesis Testing: {test_type.get().capitalize()} Test')
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Probability Density')
        canvas.draw()

    stats_frame = tk.Frame(window, bg="#333")
    stats_frame.pack(fill="x", padx=10, pady=5)

    z_label = tk.Label(stats_frame, text=f"Z-Statistic: {z:.4f}", font=("Arial", 12), bg="#333", fg="white")
    z_label.pack(side="left", padx=20)
    p_value_label = tk.Label(stats_frame, text=f"P-Value: {initial_p:.4f}", font=("Arial", 12), bg="#333", fg="white")
    p_value_label.pack(side="right", padx=20)

    update_graph()


# ─── 8) Bayes' Theorem Calculation ─────────────────────────────────────────────
def bayes_theorem():
    window = Toplevel()
    window.title("Bayes' Theorem")
    window.geometry("700x650")
    window.configure(bg="#333")

    P_B_given_A = simpledialog.askfloat("Input", "Enter P(B|A) (Likelihood of B given A):", parent=window)
    P_A = simpledialog.askfloat("Input", "Enter P(A) (Prior Probability of A):", parent=window)
    P_B_given_Ac = simpledialog.askfloat("Input", "Enter P(B|not A) (Likelihood of B given not A):", parent=window)
    if None in (P_B_given_A, P_A, P_B_given_Ac) or not (
            0 <= P_B_given_A <= 1 and 0 <= P_A <= 1 and 0 <= P_B_given_Ac <= 1):
        messagebox.showerror("Error", "All values must be entered and between 0 and 1!")
        return

    P_Ac = 1 - P_A
    P_B = (P_B_given_A * P_A) + (P_B_given_Ac * P_Ac)
    P_A_given_B = (P_B_given_A * P_A) / P_B if P_B > 0 else 0
    P_Ac_given_B = 1 - P_A_given_B

    P_A_and_B = P_B_given_A * P_A
    P_A_and_Bc = P_A - P_A_and_B
    P_Ac_and_B = P_B_given_Ac * P_Ac
    P_Ac_and_Bc = P_Ac - P_Ac_and_B

    header_frame = tk.Frame(window, bg="#444", padx=10, pady=10)
    header_frame.pack(fill="x", padx=10, pady=10)
    title_label = tk.Label(header_frame, text="Bayes' Theorem Calculator",
                           font=("Arial", 16, "bold"), bg="#444", fg="white")
    title_label.pack(pady=(0, 10))
    summary_text = f"Prior P(A): {P_A:.4f}  |  Likelihood P(B|A): {P_B_given_A:.4f}  |  P(B|not A): {P_B_given_Ac:.4f}"
    summary_label = tk.Label(header_frame, text=summary_text, font=("Arial", 11), bg="#444", fg="white")
    summary_label.pack(pady=5)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 9))

    # Joint probability heatmap
    ax1 = plt.subplot(2, 1, 1)
    data = np.array([[P_A_and_B, P_A_and_Bc], [P_Ac_and_B, P_Ac_and_Bc]])
    cmap = plt.cm.Blues
    hm = sns.heatmap(data, annot=True, fmt='.4f', cmap=cmap,
                     xticklabels=['B', 'not B'], yticklabels=['A', 'not A'], ax=ax1,
                     annot_kws={"size": 14, "weight": "bold"},
                     linewidths=1, linecolor='white',
                     cbar_kws={"shrink": 0.75, "label": "Probability"})
    labels = [['P(A∩B)', 'P(A∩not B)'], ['P(not A∩B)', 'P(not A∩not B)']]
    for i in range(2):
        for j in range(2):
            ax1.text(j + 0.5, i + 0.15, labels[i][j],
                     ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax1.set_title("Joint Probability Distribution", fontsize=14, fontweight="bold", pad=20)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(12); label.set_fontweight("bold")
    box_text = [
        f"Prior: P(A) = {P_A:.4f}",
        f"Likelihood: P(B|A) = {P_B_given_A:.4f}",
        f"P(B|not A) = {P_B_given_Ac:.4f}",
        f"Total Probability: P(B) = {P_B:.4f}",
        f"Posterior: P(A|B) = {P_A_given_B:.4f}"
    ]
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', pad=1)
    ax1.text(1.25, 0.5, '\n'.join(box_text), transform=ax1.transAxes,
             fontsize=11, verticalalignment='center', bbox=props)
    plt.subplots_adjust(hspace=0.4, right=0.85)

    # Posterior bar chart
    ax2 = plt.subplot(2, 1, 2)
    categories = ['P(A|B)', 'P(not A|B)']
    values = [P_A_given_B, P_Ac_given_B]
    bar_colors = ['#4285F4', '#EA4335']
    bars = ax2.bar(categories, values, color=bar_colors, width=0.6, edgecolor='black', linewidth=1)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='black')
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Posterior Probability Distribution", fontsize=14, fontweight="bold", pad=20)
    ax2.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for label in ax2.get_xticklabels(): label.set_fontsize(12); label.set_fontweight("bold")
    for label in ax2.get_yticklabels(): label.set_fontsize(10)

    formula_frame = tk.Frame(window, bg="#444", padx=10, pady=10)
    formula_frame.pack(fill="x", padx=10, pady=10)
    formula_text = (
        f"Bayes' Formula:  P(A|B) = [P(B|A) × P(A)] ÷ P(B) = "
        f"[{P_B_given_A:.4f} × {P_A:.4f}] ÷ {P_B:.4f} = {P_A_given_B:.4f}"
    )
    formula_label = tk.Label(formula_frame, text=formula_text,
                             font=("Arial", 13, "bold"), bg="#444", fg="white")
    formula_label.pack(pady=5)

    canvas_plot = FigureCanvasTkAgg(fig, master=window)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    save_button = tk.Button(window, text="Save Visualization", command=lambda: _save_plot(fig), font=("Arial", 12, "bold"))
    style_button(save_button)
    save_button.pack(pady=5)

    results_frame = tk.Frame(window, bg="#333")
    results_frame.pack(fill="x", padx=10, pady=10)
    results = [
        f"Prior Probability: P(A) = {P_A:.4f}",
        f"Likelihood: P(B|A) = {P_B_given_A:.4f}",
        f"Total Probability P(B): {P_B:.4f}",
        f"Posterior Probability P(A|B): {P_A_given_B:.4f}"
    ]
    for result in results:
        result_label = tk.Label(results_frame, text=result, font=("Arial", 12, "bold"), bg="#333", fg="white")
        result_label.pack(pady=2)


# ─── Main GUI Setup ────────────────────────────────────────────────────────────
def create_gui():
    root = tk.Tk()
    root.title("Statistics & Probability Calculator")
    root.geometry("650x550")
    root.configure(bg="#222")  # Dark mode

    title_label = tk.Label(
        root,
        text="Statistics & Probability Calculator",
        font=("Arial", 18, "bold"),
        bg="#222",
        fg="white"
    )
    title_label.pack(pady=20)

    subtitle_label = tk.Label(
        root,
        text="Choose a Calculation:",
        font=("Arial", 14),
        bg="#222",
        fg="white"
    )
    subtitle_label.pack(pady=10)

    buttons_frame = tk.Frame(root, bg="#222")
    buttons_frame.pack(pady=10)

    options = [
        "Dataset Visualization",
        "PDF",
        "CDF",
        "Binomial Distribution",
        "Bayes' Theorem",
        "Gamma Distribution",
        "Confidence Interval",
        "Hypothesis Testing"
    ]

    for i, opt in enumerate(options):
        btn = tk.Button(buttons_frame, text=opt, command=lambda o=opt: open_choice(o, root))
        style_button(btn)
        row, col = divmod(i, 2)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

    footer_label = tk.Label(
        root,
        text="Made with Python & Tkinter",
        font=("Arial", 10),
        bg="#222",
        fg="gray"
    )
    footer_label.pack(side="bottom", pady=15)

    root.mainloop()


def open_choice(choice, root):
    if choice == "Dataset Visualization":
        dataset_visualization(root)
    elif choice == "PDF":
        pdf_visualization(root)
    elif choice == "CDF":
        cdf_visualization(root)
    elif choice == "Binomial Distribution":
        binomial_distribution()
    elif choice == "Bayes' Theorem":
        bayes_theorem()
    elif choice == "Gamma Distribution":
        gamma_distribution()
    elif choice == "Confidence Interval":
        confidence_interval()
    elif choice == "Hypothesis Testing":
        hypothesis_testing()


if __name__ == "__main__":
    create_gui()
