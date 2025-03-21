import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, Toplevel, Scrollbar, Canvas, Frame, Text, Label, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import gamma, binom, norm as stats


# Function to Style Buttons (White with Black Text)
def style_button(btn):
    btn.config(bg="white", fg="black", font=("Arial", 12, "bold"),
               activebackground="#cccccc", activeforeground="black",
               borderwidth=2, highlightthickness=0, relief="solid",
               highlightbackground="white", highlightcolor="white",
               disabledforeground="black", takefocus=False, padx=10, pady=5)



# 📌 Dataset Visualization Window (With Scrollbar for Stats)
def dataset_visualization(root):
    window = Toplevel(root)
    window.title("Dataset Visualization")
    window.geometry("1250x900")
    window.configure(bg="#333")

    # 🟢 Scrollable Canvas
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

    # 🟢 User Input for Dataset
    x_input = simpledialog.askstring("Input", "Enter X values (comma-separated):", parent=window)
    y_input = simpledialog.askstring("Input", "Enter Y values (comma-separated):", parent=window)

    try:
        x_values = list(map(float, x_input.split(",")))
        y_values = list(map(float, y_input.split(",")))

        if len(x_values) != len(y_values):
            messagebox.showerror("Error", "X and Y values must have the same length!")
            return

        df = pd.DataFrame({'x': x_values, 'y': y_values})
        visualize_data(df, frame, canvas)

    except ValueError:
        messagebox.showerror("Error", "Invalid numeric input! Please enter numbers separated by commas.")


# 📌 Data Visualization Function (Includes All Stats & Scrollable)
def visualize_data(df, frame, canvas):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.set_style("darkgrid")

    # 🟢 Plots
    sns.scatterplot(x=df['x'], y=df['y'], ax=axes[0, 0])
    sns.lineplot(x=df['x'], y=df['y'], ax=axes[0, 1], marker='o')
    sns.boxplot(y=df['y'], ax=axes[1, 0], showmeans=True, meanline=True)
    sns.histplot(df['y'], kde=True, ax=axes[1, 1])

    plt.tight_layout()

    # 🟢 Embed Matplotlib Plot in Tkinter
    canvas_plot = FigureCanvasTkAgg(fig, master=frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack()

    # 📌 Compute Descriptive Statistics
    stats_info = df.describe().to_string()

    # 📌 Scrollable Text Widget for Stats
    text_frame = Frame(frame, bg="#333")
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)

    text_widget = Text(text_frame, height=12, wrap="none", bg="#333", fg="white", font=("Arial", 12))
    text_widget.insert("1.0", stats_info)
    text_widget.config(state="disabled")

    scroll_y = Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    text_widget.pack(fill="both", expand=True)




# Binomial Distribution Plot (With Dynamic Coloring)
def plot_binomial_graph(n, p, x, selected_type, parent_frame):
    for widget in parent_frame.winfo_children():
        widget.destroy()

    x_vals = np.arange(0, n + 1)
    y_vals = binom.pmf(x_vals, n, p)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["red" if (selected_type == "P(X = x)" and val == x) or
                       (selected_type == "P(X ≤ x)" and val <= x) or
                       (selected_type == "P(X ≥ x)" and val >= x) else "gray" for val in x_vals]

    ax.bar(x_vals, y_vals, color=colors, alpha=0.7, edgecolor="black")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


# Binomial Distribution Calculator
def binomial_distribution():
    window = Toplevel()
    window.title("Binomial Distribution")
    window.geometry("600x500")
    window.configure(bg="#333")

    # Input fields for binomial parameters
    n = simpledialog.askinteger("Input", "Enter number of trials (n):", parent=window)
    p = simpledialog.askfloat("Input", "Enter probability of success (p):", parent=window)
    x = simpledialog.askinteger("Input", "Enter value to calculate (x):", parent=window)

    if n is None or p is None or x is None or n <= 0 or p < 0 or p > 1 or x < 0 or x > n:
        messagebox.showerror("Error",
                             "Invalid parameters! n must be positive, p must be between 0 and 1, and x must be between 0 and n.")
        return

    # Frame for plot
    plot_frame = tk.Frame(window, bg="#333")
    plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Create buttons for different probability types
    prob_types = ["P(X = x)", "P(X ≤ x)", "P(X ≥ x)"]
    buttons_frame = tk.Frame(window, bg="#333")
    buttons_frame.pack(fill="x", padx=10, pady=5)

    for prob_type in prob_types:
        btn = tk.Button(buttons_frame, text=prob_type,
                        command=lambda t=prob_type: update_prob_display(t))
        style_button(btn)
        btn.pack(side="left", padx=5, expand=True)

    # Function to update probability display
    def update_prob_display(selected_type):
        if selected_type == "P(X = x)":
            prob = binom.pmf(x, n, p)
        elif selected_type == "P(X ≤ x)":
            prob = binom.cdf(x, n, p)
        else:  # P(X ≥ x)
            prob = 1 - binom.cdf(x - 1, n, p)

        result_label.config(text=f"{selected_type} = {prob:.6f}")
        plot_binomial_graph(n, p, x, selected_type, plot_frame)

    # Result label
    result_label = tk.Label(window, text="", font=("Arial", 14), bg="#333", fg="white")
    result_label.pack(pady=10)

    # Initial plot with P(X = x)
    update_prob_display("P(X = x)")


# Gamma Distribution Calculator
def gamma_distribution():
    window = Toplevel()
    window.title("Gamma Distribution")
    window.geometry("600x500")
    window.configure(bg="#333")

    # Input fields for shape and scale parameters
    shape = simpledialog.askfloat("Input", "Enter shape parameter (k):", parent=window)
    scale = simpledialog.askfloat("Input", "Enter scale parameter (θ):", parent=window)

    if shape is None or scale is None or shape <= 0 or scale <= 0:
        messagebox.showerror("Error", "Shape and Scale must be positive numbers!")
        return

    # Generate the x and y values for the plot
    x_vals = np.linspace(0, 20, 1000)
    y_vals = gamma.pdf(x_vals, a=shape, scale=scale)

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, y_vals, label=f'Gamma PDF (k={shape}, θ={scale})')
    ax.set_title("Gamma Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.legend()

    # Render the plot onto the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Function to save the plot as an image
    def save_plot():
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            fig.savefig(file_path)
            messagebox.showinfo("Saved", f"Plot saved as {file_path}")

    # Create a Save button
    save_button = tk.Button(window, text="Save Plot", command=save_plot)
    save_button.pack(pady=10)

    # Function to update the coordinates displayed in real-time
    def update_coordinates(event):
        # Get the mouse position (relative to the canvas widget)
        x = event.x
        y = event.y

        # Convert to data coordinates
        canvas_width = canvas.get_tk_widget().winfo_width()
        canvas_height = canvas.get_tk_widget().winfo_height()

        data_x = ax.get_xlim()[0] + (x / canvas_width) * (ax.get_xlim()[1] - ax.get_xlim()[0])
        data_y = ax.get_ylim()[0] + ((canvas_height - y) / canvas_height) * (ax.get_ylim()[1] - ax.get_ylim()[0])

        # Update the coordinates label
        coordinates_label.config(text=f"X: {data_x:.2f}, Y: {data_y:.2f}")

    # Label for displaying the coordinates
    coordinates_label = tk.Label(window, text="X: 0.00, Y: 0.00", font=("Arial", 12), bg="#333", fg="white")
    coordinates_label.pack(pady=5)

    # Bind mouse motion event to the plot canvas widget
    canvas.get_tk_widget().bind("<Motion>", update_coordinates)


# Confidence Interval Calculation
def confidence_interval():
    window = Toplevel()
    window.title("Confidence Interval")
    window.geometry("600x500")
    window.configure(bg="#333")

    # Input Fields for CI
    mean = simpledialog.askfloat("Input", "Enter the sample mean (x̄):", parent=window)
    std_dev = simpledialog.askfloat("Input", "Enter the sample standard deviation (s):", parent=window)
    n = simpledialog.askinteger("Input", "Enter the sample size (n):", parent=window)
    confidence_level = simpledialog.askfloat("Input", "Enter the confidence level (e.g., 0.95 for 95%):", parent=window)

    if mean is None or std_dev is None or n is None or confidence_level is None:
        messagebox.showerror("Error", "All values must be entered!")
        return

    # Calculate Standard Error
    SE = std_dev / np.sqrt(n)

    # Z-value for Confidence Level
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence_level)

    if not z:
        messagebox.showerror("Error", "Invalid confidence level! Please use 0.90, 0.95, or 0.99")
        return

    # Confidence Interval Calculation
    margin_of_error = z * SE
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    # Create visualization of confidence interval
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

    # Display the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Display Results as text
    result_label = tk.Label(window, text=f"Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]",
                            font=("Arial", 14), bg="#333", fg="white")
    result_label.pack(pady=10)



# Confidence Interval Calculation
def confidence_interval():
    window = Toplevel()
    window.title("Confidence Interval")
    window.geometry("600x500")
    window.configure(bg="#333")

    # Input Fields for CI
    mean = simpledialog.askfloat("Input", "Enter the sample mean (x̄):", parent=window)
    std_dev = simpledialog.askfloat("Input", "Enter the sample standard deviation (s):", parent=window)
    n = simpledialog.askinteger("Input", "Enter the sample size (n):", parent=window)
    confidence_level = simpledialog.askfloat("Input", "Enter the confidence level (e.g., 0.95 for 95%):", parent=window)

    if mean is None or std_dev is None or n is None or confidence_level is None:
        messagebox.showerror("Error", "All values must be entered!")
        return

    # Calculate Standard Error
    SE = std_dev / np.sqrt(n)

    # Z-value for Confidence Level
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence_level)

    if not z:
        messagebox.showerror("Error", "Invalid confidence level! Please use 0.90, 0.95, or 0.99")
        return

    # Confidence Interval Calculation
    margin_of_error = z * SE
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    # Create visualization of confidence interval
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

    # Display the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Display Results as text
    result_label = tk.Label(window, text=f"Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]",
                            font=("Arial", 14), bg="#333", fg="white")
    result_label.pack(pady=10)


# Hypothesis Testing Calculation
def hypothesis_testing():
    window = Toplevel()
    window.title("Hypothesis Testing")
    window.geometry("700x600")
    window.configure(bg="#333")

    # Input Fields for Hypothesis Test
    sample_mean = simpledialog.askfloat("Input", "Enter the sample mean (x̄):", parent=window)
    population_mean = simpledialog.askfloat("Input", "Enter the population mean (μ):", parent=window)
    std_dev = simpledialog.askfloat("Input", "Enter the population standard deviation (σ):", parent=window)
    n = simpledialog.askinteger("Input", "Enter the sample size (n):", parent=window)

    if sample_mean is None or population_mean is None or std_dev is None or n is None:
        messagebox.showerror("Error", "All values must be entered!")
        return

    # Calculate Standard Error
    SE = std_dev / np.sqrt(n)

    # Calculate Z-Statistic
    z = (sample_mean - population_mean) / SE

    # Initialize Test Type
    test_type = StringVar(value="two-tailed")  # Default to two-tailed

    # Create frames for controls and graph
    controls_frame = tk.Frame(window, bg="#333")
    controls_frame.pack(fill="x", padx=10, pady=10)

    graph_frame = tk.Frame(window, bg="#333")
    graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Create Test Type Toggle Button
    def toggle_test_type():
        if test_type.get() == "two-tailed":
            test_type.set("one-tailed")
            toggle_button.config(text="Switch to Two-Tailed Test")
        else:
            test_type.set("two-tailed")
            toggle_button.config(text="Switch to One-Tailed Test")
        update_graph()

    toggle_button = tk.Button(controls_frame, text="Switch to One-Tailed Test",
                              command=toggle_test_type, font=("Arial", 12, "bold"))
    style_button(toggle_button)
    toggle_button.pack(side="left", padx=5)

    # Calculate P-Value based on Test Type
    def calculate_p_value(z, test_type):
        if test_type == "two-tailed":
            return 2 * (1 - stats.cdf(abs(z)))  # Two-tailed p-value
        else:  # One-tailed
            return 1 - stats.cdf(z) if z > 0 else stats.cdf(z)

    # Calculate initial p-value
    p_value = calculate_p_value(z, test_type.get())

    # Create Graph for Hypothesis Test
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Plot the standard normal distribution curve
    x = np.linspace(-4, 4, 1000)
    y = stats.pdf(x)

    # Define the Graph Update Function
    def update_graph():
        ax.clear()  # Clear the previous graph

        # Recalculate p-value with current test type
        new_p_value = calculate_p_value(z, test_type.get())
        p_value_label.config(text=f"P-Value: {new_p_value:.4f}")

        # Plot normal distribution
        ax.plot(x, y, label='Standard Normal Distribution', color='blue')

        # Highlight the rejection region based on the test type
        if test_type.get() == "two-tailed":
            rejection_x = x[np.where((x >= abs(z)) | (x <= -abs(z)))]
            rejection_y = stats.pdf(rejection_x)
            ax.fill_between(rejection_x, 0, rejection_y, color='red', alpha=0.3,
                            label="Rejection Region (Two-Tailed)")
        else:
            if z > 0:
                rejection_x = x[np.where(x >= z)]
                rejection_y = stats.pdf(rejection_x)
                ax.fill_between(rejection_x, 0, rejection_y, color='red', alpha=0.3,
                                label="Rejection Region (One-Tailed, Right)")
            else:
                rejection_x = x[np.where(x <= z)]
                rejection_y = stats.pdf(rejection_x)
                ax.fill_between(rejection_x, 0, rejection_y, color='red', alpha=0.3,
                                label="Rejection Region (One-Tailed, Left)")

        ax.axvline(x=z, color='black', linestyle='--', label=f'Z = {z:.4f}')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, stats.pdf(0) * 1.1)
        ax.legend()
        ax.set_title(f'Hypothesis Testing: {test_type.get().capitalize()} Test')
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Probability Density')

        canvas.draw()

    # Display Z-Statistic and P-Value
    stats_frame = tk.Frame(window, bg="#333")
    stats_frame.pack(fill="x", padx=10, pady=5)

    z_label = tk.Label(stats_frame, text=f"Z-Statistic: {z:.4f}",
                       font=("Arial", 12), bg="#333", fg="white")
    z_label.pack(side="left", padx=20)

    p_value_label = tk.Label(stats_frame, text=f"P-Value: {p_value:.4f}",
                             font=("Arial", 12), bg="#333", fg="white")
    p_value_label.pack(side="right", padx=20)

    # Initial Graph Display
    update_graph()

# Bayes' Theorem Calculation
def bayes_theorem():
    window = Toplevel()
    window.title("Bayes' Theorem")
    window.geometry("700x650")
    window.configure(bg="#333")

    # Getting input values
    P_B_given_A = simpledialog.askfloat("Input", "Enter P(B|A) (Likelihood of B given A):", parent=window)
    P_A = simpledialog.askfloat("Input", "Enter P(A) (Prior Probability of A):", parent=window)
    P_B_given_Ac = simpledialog.askfloat("Input", "Enter P(B|not A) (Likelihood of B given not A):", parent=window)

    if None in (P_B_given_A, P_A, P_B_given_Ac) or not (
            0 <= P_B_given_A <= 1 and 0 <= P_A <= 1 and 0 <= P_B_given_Ac <= 1):
        messagebox.showerror("Error", "All values must be entered and between 0 and 1!")
        return

    # Calculate Bayes' theorem components
    P_Ac = 1 - P_A
    P_B = (P_B_given_A * P_A) + (P_B_given_Ac * P_Ac)
    P_A_given_B = (P_B_given_A * P_A) / P_B if P_B > 0 else 0
    P_Ac_given_B = 1 - P_A_given_B

    # Create joint probabilities
    P_A_and_B = P_B_given_A * P_A
    P_A_and_Bc = P_A - P_A_and_B
    P_Ac_and_B = P_B_given_Ac * P_Ac
    P_Ac_and_Bc = P_Ac - P_Ac_and_B

    # Create a header frame with summary information
    header_frame = tk.Frame(window, bg="#444", padx=10, pady=10)
    header_frame.pack(fill="x", padx=10, pady=10)

    # Add a title
    title_label = tk.Label(header_frame, text="Bayes' Theorem Calculator",
                           font=("Arial", 16, "bold"), bg="#444", fg="white")
    title_label.pack(pady=(0, 10))

    # Add summary of inputs
    summary_text = f"Prior P(A): {P_A:.4f}  |  Likelihood P(B|A): {P_B_given_A:.4f}  |  P(B|not A): {P_B_given_Ac:.4f}"
    summary_label = tk.Label(header_frame, text=summary_text,
                             font=("Arial", 11), bg="#444", fg="white")
    summary_label.pack(pady=5)

    # Apply a clean style for plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a figure with two subplots, more vertical space for clarity
    fig = plt.figure(figsize=(10, 9))  # Wider figure to accommodate the box on the right

    # First subplot: Joint probability heatmap
    ax1 = plt.subplot(2, 1, 1)

    # Set up the 2x2 grid for visualization
    data = np.array([[P_A_and_B, P_A_and_Bc],
                     [P_Ac_and_B, P_Ac_and_Bc]])

    # Create a better colormap with clear contrast
    cmap = plt.cm.Blues

    # Create a cleaner heatmap with better formatting
    hm = sns.heatmap(data, annot=True, fmt='.4f', cmap=cmap,
                     xticklabels=['B', 'not B'],
                     yticklabels=['A', 'not A'],
                     ax=ax1,
                     annot_kws={"size": 14, "weight": "bold"},
                     linewidths=1, linecolor='white',
                     cbar_kws={"shrink": 0.75, "label": "Probability"})

    # Add clear labels for each quadrant
    labels = [['P(A∩B)', 'P(A∩not B)'],
              ['P(not A∩B)', 'P(not A∩not B)']]

    for i in range(2):
        for j in range(2):
            ax1.text(j + 0.5, i + 0.15, labels[i][j],
                     ha="center", va="center", color="white",
                     fontsize=11, fontweight="bold")

    ax1.set_title("Joint Probability Distribution", fontsize=14, fontweight="bold", pad=20)

    # Make tick labels more readable
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")

    # Add key information in a clean side box
    box_text = [
        f"Prior: P(A) = {P_A:.4f}",
        f"Likelihood: P(B|A) = {P_B_given_A:.4f}",
        f"P(B|not A) = {P_B_given_Ac:.4f}",
        f"Total Probability: P(B) = {P_B:.4f}",
        f"Posterior: P(A|B) = {P_A_given_B:.4f}"
    ]

    # Position the box to the right with more space
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', pad=1)
    ax1.text(1.25, 0.5, '\n'.join(box_text), transform=ax1.transAxes,
             fontsize=11, verticalalignment='center', bbox=props)

    # More space between subplots and more right margin for the info box
    plt.subplots_adjust(hspace=0.4, right=0.85)

    # Second subplot: Posterior probability visualization
    ax2 = plt.subplot(2, 1, 2)

    # Create bar chart for posterior probabilities with better colors
    categories = ['P(A|B)', 'P(not A|B)']
    values = [P_A_given_B, P_Ac_given_B]

    # Use distinct colors that are visually appealing
    bar_colors = ['#4285F4', '#EA4335']

    bars = ax2.bar(categories, values, color=bar_colors, width=0.6,
                   edgecolor='black', linewidth=1)

    # Add value labels with better formatting
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='black')

    # Improve axis formatting
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Posterior Probability Distribution", fontsize=14, fontweight="bold", pad=20)
    ax2.set_ylabel("Probability", fontsize=12, fontweight="bold")

    # Add clearer grid lines
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Make tick labels more readable
    for label in ax2.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")
    for label in ax2.get_yticklabels():
        label.set_fontsize(10)

    # Add a clear formula section
    formula_frame = tk.Frame(window, bg="#444", padx=10, pady=10)
    formula_frame.pack(fill="x", padx=10, pady=10)

    # Add the formula as a label (more reliable than matplotlib text)
    formula_text = f"Bayes' Formula:  P(A|B) = [P(B|A) × P(A)] ÷ P(B) = [{P_B_given_A:.4f} × {P_A:.4f}] ÷ {P_B:.4f} = {P_A_given_B:.4f}"
    formula_label = tk.Label(formula_frame, text=formula_text,
                             font=("Arial", 13, "bold"), bg="#444", fg="white")
    formula_label.pack(pady=5)

    # Display the visualization
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Create save function
    def save_plot():
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("PDF files", "*.pdf"),
                                                            ("SVG files", "*.svg"),
                                                            ("All files", "*.*")])
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Bayes' Theorem visualization saved as {file_path}")

    # Add save button
    save_button = tk.Button(window, text="Save Visualization", command=save_plot)
    style_button(save_button)
    save_button.pack(pady=5)

    # Display detailed results in a cleaner frame
    results_frame = tk.Frame(window, bg="#333")
    results_frame.pack(fill="x", padx=10, pady=10)

    # Create labels for each calculation result with better formatting
    results = [
        f"Prior Probability: P(A) = {P_A:.4f}",
        f"Likelihood: P(B|A) = {P_B_given_A:.4f}",
        f"Alt. Likelihood: P(B|not A) = {P_B_given_Ac:.4f}",
        f"Total Probability: P(B) = {P_B:.4f}",
        f"Posterior Probability: P(A|B) = {P_A_given_B:.4f}"
    ]

    for result in results:
        result_label = tk.Label(results_frame, text=result, font=("Arial", 12, "bold"),
                                bg="#333", fg="white")
        result_label.pack(pady=2)


# Main GUI Setup Function
def create_gui():
    root = tk.Tk()
    root.title("Statistics & Probability Calculator")
    root.geometry("650x450")
    root.configure(bg="#222")  # Dark mode

    title_label = tk.Label(root, text="Statistics & Probability Calculator",
                           font=("Arial", 18, "bold"), bg="#222", fg="white")
    title_label.pack(pady=20)

    subtitle_label = tk.Label(root, text="Choose a Calculation:",
                              font=("Arial", 14), bg="#222", fg="white")
    subtitle_label.pack(pady=10)

    # Create a frame for the buttons
    buttons_frame = tk.Frame(root, bg="#222")
    buttons_frame.pack(pady=10)

    options = ["Dataset Visualization", "Binomial Distribution", "Bayes' Theorem",
               "Gamma Distribution", "Confidence Interval", "Hypothesis Testing"]

    # Organize buttons in a grid (2 columns)
    for i, opt in enumerate(options):
        btn = tk.Button(buttons_frame, text=opt, command=lambda o=opt: open_choice(o, root))
        style_button(btn)
        row, col = divmod(i, 2)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

    # Add a footer
    footer_label = tk.Label(root, text="Made with Python & Tkinter",
                            font=("Arial", 10), bg="#222", fg="gray")
    footer_label.pack(side="bottom", pady=15)

    root.mainloop()


# Handle Option Selection
def open_choice(choice, root):
    if choice == "Dataset Visualization":
        dataset_visualization(root)
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


# Run the application
if __name__ == "__main__":
    create_gui()




# Bayes' Theorem Calculation
def bayes_theorem():
    window = Toplevel()
    window.title("Bayes' Theorem")
    window.geometry("700x650")
    window.configure(bg="#333")

    # Getting input values
    P_B_given_A = simpledialog.askfloat("Input", "Enter P(B|A) (Likelihood of B given A):", parent=window)
    P_A = simpledialog.askfloat("Input", "Enter P(A) (Prior Probability of A):", parent=window)
    P_B_given_Ac = simpledialog.askfloat("Input", "Enter P(B|A^c) (Likelihood of B given not A):", parent=window)

    if None in (P_B_given_A, P_A, P_B_given_Ac) or not (
            0 <= P_B_given_A <= 1 and 0 <= P_A <= 1 and 0 <= P_B_given_Ac <= 1):
        messagebox.showerror("Error", "All values must be entered and between 0 and 1!")
        return

    # Calculate Bayes' theorem components
    P_Ac = 1 - P_A
    P_B = (P_B_given_A * P_A) + (P_B_given_Ac * P_Ac)
    P_A_given_B = (P_B_given_A * P_A) / P_B if P_B > 0 else 0
    P_Ac_given_B = 1 - P_A_given_B

    # Create joint probabilities
    P_A_and_B = P_B_given_A * P_A
    P_A_and_Bc = P_A - P_A_and_B
    P_Ac_and_B = P_B_given_Ac * P_Ac
    P_Ac_and_Bc = P_Ac - P_Ac_and_B

    # Create a figure with two subplots
    fig = plt.figure(figsize=(10, 8))

    # First subplot: Joint probability heatmap
    ax1 = plt.subplot(2, 1, 1)

    # Set up the 2x2 grid for visualization
    data = np.array([[P_A_and_B, P_A_and_Bc],
                     [P_Ac_and_B, P_Ac_and_Bc]])

    # Create the heatmap
    hm = sns.heatmap(data, annot=True, fmt='.4f', cmap='Blues',
                     xticklabels=['B', 'not B'],
                     yticklabels=['A', 'not A'], ax=ax1)

    ax1.set_title("Joint Probability Distribution")

    # Annotations for Bayes' Theorem components
    textstr = '\n'.join((
        f"P(A) (Prior): {P_A:.4f}",
        f"P(B|A) (Likelihood): {P_B_given_A:.4f}",
        f"P(B) (Total Probability): {P_B:.4f}",
        f"P(A|B) (Posterior): {P_A_given_B:.4f}"
    ))

    # Add text box with calculations
    ax1.text(1.05, 0.5, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Second subplot: Posterior probability visualization
    ax2 = plt.subplot(2, 1, 2)

    # Create bar chart for posterior probabilities
    categories = ['P(A|B)', 'P(not A|B)']
    values = [P_A_given_B, P_Ac_given_B]

    bars = ax2.bar(categories, values, color=['green', 'red'])

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom')

    ax2.set_ylim(0, 1.1)
    ax2.set_title("Posterior Probability Distribution")
    ax2.set_ylabel("Probability")

    # Add Bayes' formula as text
    formula = r"$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$"
    calculation = f"= \\frac{{{P_B_given_A:.4f} \\cdot {P_A:.4f}}}{{{P_B:.4f}}} = {P_A_given_B:.4f}"

    ax2.text(0.5, -0.25, formula + "\n" + calculation,
             transform=ax2.transAxes, fontsize=12, ha='center')

    plt.tight_layout()

    # Display the visualization
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Create save function
    def save_plot():
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("PDF files", "*.pdf"),
                                                            ("All files", "*.*")])
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Bayes' Theorem visualization saved as {file_path}")

    # Add save button
    save_button = tk.Button(window, text="Save Visualization", command=save_plot)
    style_button(save_button)
    save_button.pack(pady=5)

    # Display detailed results text
    results_frame = tk.Frame(window, bg="#333")
    results_frame.pack(fill="x", padx=10, pady=10)

    # Create labels for each calculation result
    results = [
        f"Prior Probability P(A): {P_A:.4f}",
        f"Likelihood P(B|A): {P_B_given_A:.4f}",
        f"Total Probability P(B): {P_B:.4f}",
        f"Posterior Probability P(A|B): {P_A_given_B:.4f}"
    ]

    for result in results:
        result_label = tk.Label(results_frame, text=result, font=("Arial", 12), bg="#333", fg="white")
        result_label.pack(pady=2)


# Main GUI Setup Function
def create_gui():
    root = tk.Tk()
    root.title("Statistics & Probability Calculator")
    root.geometry("650x450")
    root.configure(bg="#222")  # Dark mode

    title_label = tk.Label(root, text="Statistics & Probability Calculator",
                           font=("Arial", 18, "bold"), bg="#222", fg="white")
    title_label.pack(pady=20)

    subtitle_label = tk.Label(root, text="Choose a Calculation:",
                              font=("Arial", 14), bg="#222", fg="white")
    subtitle_label.pack(pady=10)

    # Create a frame for the buttons
    buttons_frame = tk.Frame(root, bg="#222")
    buttons_frame.pack(pady=10)

    options = ["Dataset Visualization", "Binomial Distribution", "Bayes' Theorem",
               "Gamma Distribution", "Confidence Interval", "Hypothesis Testing"]

    # Organize buttons in a grid (2 columns)
    for i, opt in enumerate(options):
        btn = tk.Button(buttons_frame, text=opt, command=lambda o=opt: open_choice(o, root))
        style_button(btn)
        row, col = divmod(i, 2)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

    # Add a footer
    footer_label = tk.Label(root, text="Made with Python & Tkinter",
                            font=("Arial", 10), bg="#222", fg="gray")
    footer_label.pack(side="bottom", pady=15)

    root.mainloop()


# Handle Option Selection
def open_choice(choice, root):
    if choice == "Dataset Visualization":
        dataset_visualization(root)
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


# Run the application
if __name__ == "__main__":
    create_gui()
