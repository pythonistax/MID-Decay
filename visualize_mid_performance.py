import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
pooled_df = pd.read_csv('pooled_example.csv')
pooled_df['date'] = pd.to_datetime(pooled_df['date'])

# Get unique MIDs
unique_mids = sorted(pooled_df['mid'].unique())

# Create figure for each MID
for mid in unique_mids:
    # Filter data for this MID
    mid_data = pooled_df[pooled_df['mid'] == mid].sort_values('date')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main title
    fig.suptitle(f'MID {mid} - Performance Analysis Dashboard', fontsize=20, fontweight='bold', y=1.02)
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Success Rate Over Time (Main Chart)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(mid_data['date'], mid_data['success_charge_ratess'], 
             linewidth=3, marker='o', markersize=6, color='#2E86AB', label='Daily Rate')
    
    # Add moving average
    if len(mid_data) > 7:
        ma7 = mid_data['success_charge_ratess'].rolling(window=7, min_periods=1).mean()
        ax1.plot(mid_data['date'], ma7, linewidth=2, alpha=0.7, 
                color='#A23B72', linestyle='--', label='7-Day MA')
    
    # Add trend line
    z = np.polyfit(range(len(mid_data)), mid_data['success_charge_ratess'], 1)
    p = np.poly1d(z)
    ax1.plot(mid_data['date'], p(range(len(mid_data))), 
            linewidth=2, alpha=0.5, color='#F18F01', linestyle=':', label='Trend')
    
    ax1.set_title('Success Charge Rate Over Time', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', frameon=True, shadow=True)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add baseline reference
    ax1.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.text(mid_data['date'].iloc[0], 0.52, 'Baseline (50%)', fontsize=9, alpha=0.6)
    
    # 2. Revenue Performance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(mid_data['date'], mid_data['net_revenue'], 
           color='#27AE60', alpha=0.7, edgecolor='darkgreen', linewidth=1)
    ax2.set_title('Daily Net Revenue', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Revenue ($)', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Volume Metrics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(mid_data['date'], mid_data['attempted_charges'], 
            marker='s', markersize=5, linewidth=2, label='Attempted', color='#3498DB')
    ax3.plot(mid_data['date'], mid_data['successful_charges'], 
            marker='^', markersize=5, linewidth=2, label='Successful', color='#2ECC71')
    ax3.plot(mid_data['date'], mid_data['declined_charges'], 
            marker='v', markersize=5, linewidth=2, label='Declined', color='#E74C3C')
    ax3.set_title('Charge Volume Breakdown', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Number of Charges', fontsize=10)
    ax3.legend(loc='best', frameon=True, shadow=True, fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Cumulative Performance
    ax4 = fig.add_subplot(gs[2, 0])
    cumulative_revenue = mid_data['net_revenue'].cumsum()
    ax4.fill_between(mid_data['date'], 0, cumulative_revenue, 
                    alpha=0.4, color='#9B59B6', edgecolor='#8E44AD', linewidth=2)
    ax4.set_title('Cumulative Net Revenue', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Cumulative Revenue ($)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Key Metrics Summary Box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate metrics
    avg_success_rate = mid_data['success_charge_ratess'].mean()
    total_revenue = mid_data['net_revenue'].sum()
    total_attempts = mid_data['attempted_charges'].sum()
    days_active = (mid_data['date'].max() - mid_data['date'].min()).days + 1
    peak_rate = mid_data['success_charge_ratess'].max()
    current_rate = mid_data['success_charge_ratess'].iloc[-1]
    
    # Determine trend
    if len(mid_data) > 1:
        recent_trend = "↑" if mid_data['success_charge_ratess'].iloc[-1] > mid_data['success_charge_ratess'].iloc[-5:].mean() else "↓"
    else:
        recent_trend = "→"
    
    # Create summary text
    summary_text = f"""
    📊 KEY PERFORMANCE METRICS
    
    Average Success Rate: {avg_success_rate:.1%}
    Current Success Rate: {current_rate:.1%} {recent_trend}
    Peak Success Rate: {peak_rate:.1%}
    
    Total Revenue: ${total_revenue:,.2f}
    Total Attempts: {total_attempts:,}
    Days Active: {days_active}
    
    Daily Avg Revenue: ${total_revenue/max(days_active,1):.2f}
    Revenue per Attempt: ${total_revenue/max(total_attempts,1):.2f}
    """
    
    # Style the text box
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.2, edgecolor="gray")
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', bbox=bbox_props, family='monospace')
    
    # Add performance indicator
    if current_rate >= 0.5:
        status = "🟢 HEALTHY"
        color = "green"
    elif current_rate >= 0.3:
        status = "🟡 MONITOR"
        color = "orange"
    else:
        status = "🔴 CRITICAL"
        color = "red"
    
    ax5.text(0.5, 0.15, status, transform=ax5.transAxes, fontsize=16,
            horizontalalignment='center', fontweight='bold', color=color)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed visualization for MID {mid}")
    print("-" * 50)