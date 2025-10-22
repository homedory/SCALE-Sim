#!/usr/bin/env python3
"""
AlexNet Energy Consumption Visualization
Reads energy CSV files and creates bar charts showing energy breakdown by component
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_energy_csv(filepath):
    """Read energy CSV file and return parsed data"""
    df = pd.read_csv(filepath)
    # Remove the TOTAL row
    df = df[df['Component'] != 'TOTAL']
    return df

def group_components(df):
    """Group components into 5 main categories"""
    grouped_data = {
        'DRAM': 0,
        'SRAM': 0, 
        'PE_Transfer': 0,
        'RF': 0,
        'ALU': 0
    }
    
    for _, row in df.iterrows():
        component = row['Component']
        energy = row['Total_Energy_pJ']
        
        if 'DRAM' in component:
            grouped_data['DRAM'] += energy
        elif 'SRAM' in component:
            grouped_data['SRAM'] += energy
        elif 'PE_Data_Transfer' in component:
            grouped_data['PE_Transfer'] += energy
        elif 'RF' in component:
            grouped_data['RF'] += energy
        elif 'ALU' in component:
            grouped_data['ALU'] += energy
    
    return grouped_data

def print_layer_energy_breakdown(layer_name, df):
    """Print detailed energy breakdown for a layer in the specified format"""
    print("=" * 60)
    print(f"{layer_name.upper()} ENERGY CONSUMPTION BREAKDOWN")
    print("=" * 60)
    
    # Energy per operation values (matching the format)
    energy_per_op = {
        'DRAM_Read': 500.0,
        'DRAM_Write': 500.0,
        'SRAM_Read': 10.0,
        'SRAM_Write': 10.0,
        'PE_Data_Transfer': 3.0,
        'RF_Read': 1.0,
        'RF_Write': 1.0,
        'ALU_Operations': 1.0
    }
    
    total_energy = 0
    
    for _, row in df.iterrows():
        component = row['Component']
        operation_count = int(row['Operation_Count'])
        energy_per_op_val = row['Energy_per_Op_pJ']
        total_energy_val = row['Total_Energy_pJ']
        
        # Format the component name for display
        display_name = component.replace('_', ' ')
        
        print(f"{display_name:<25}: {operation_count:>10} x {energy_per_op_val:>6.1f} pJ = {total_energy_val:>12.1f} pJ")
        total_energy += total_energy_val
    
    print("-" * 60)
    print(f"TOTAL ENERGY: {total_energy:>48.1f} pJ")
    print("=" * 60)
    print()  # Add blank line after each layer

def plot_energy_values(layer_data, total_network_energy):
    """Plot energy consumption values (not percentages) for each layer and total"""
    
    # Create figure with 3 subplots (2 graphs + 1 table)
    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Define colors and components
    components = ['DRAM', 'SRAM', 'PE_Transfer', 'RF', 'ALU']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    layers = list(layer_data.keys())
    
    # Plot 1: Layer-wise energy values (in μJ)
    x_pos = np.arange(len(layers))
    width = 0.15  # Width of bars
    
    for i, component in enumerate(components):
        values = [layer_data[layer][component] / 1e6 for layer in layers]  # Convert pJ to μJ
        offset = (i - 2) * width  # Center the bars
        bars = ax1.bar(x_pos + offset, values, width, label=component, color=colors[i])
        
        # Add value labels on bars (only if value > 1 μJ to avoid clutter)
        for bar, value in zip(bars, values):
            if value > 1:
                height = bar.get_height()
                # Position label inside the bar for tall bars, outside for short bars
                if height > max(values) * 0.1:  # If bar is tall enough
                    ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                            f'{value:.1f}', ha='center', va='center', fontsize=8, 
                            rotation=90, color='black', fontweight='bold')
                else:
                    max_val = max([layer_data[layer][comp] / 1e6 for layer in layers for comp in components])
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.05,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8, 
                            rotation=90, color='black', fontweight='bold')
    
    ax1.set_title('Energy Consumption by Layer (μJ)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy (μJ)', fontsize=12)
    ax1.set_xlabel('AlexNet Layers', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layers)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    # Set appropriate y-axis range based on data
    min_val = min([v for v in [layer_data[layer][comp] / 1e6 for layer in layers for comp in components] if v > 0])
    max_val = max([layer_data[layer][comp] / 1e6 for layer in layers for comp in components])
    ax1.set_ylim(bottom=0, top=max_val * 1.2)  # Show from 0 to 120% of max value
    
    # Plot 2: Total network energy values (in μJ)
    total_values = [total_network_energy[comp] / 1e6 for comp in components]  # Convert pJ to μJ
    bars = ax2.bar(components, total_values, color=colors)
    
    ax2.set_title('Total AlexNet Energy Consumption (μJ)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy (μJ)', fontsize=12)
    ax2.set_xlabel('Components', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # Set appropriate y-axis range based on data
    ax2.set_ylim(bottom=0, top=max(total_values) * 1.2)  # Show from 0 to 120% of max value
    
    # Add value labels on bars
    for bar, value in zip(bars, total_values):
        height = bar.get_height()
        # Position label inside the bar for tall bars
        if height > max(total_values) * 0.1:  # If bar is tall enough
            ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{value:.1f} μJ', ha='center', va='center', fontweight='bold', 
                    fontsize=11, color='black')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(total_values) * 0.05,
                    f'{value:.1f} μJ', ha='center', va='bottom', fontweight='bold', 
                    fontsize=11, color='black')
    
    # Plot 3: Total Energy Summary Table
    ax3.axis('off')  # Turn off axis for table
    
    # Create table data
    total_network_energy_value = sum(total_network_energy.values()) / 1e6
    table_data = [
        ['Component', 'Energy (μJ)'],
        ['DRAM', f'{total_values[0]:.1f}'],
        ['SRAM', f'{total_values[1]:.1f}'],
        ['PE_Transfer', f'{total_values[2]:.1f}'],
        ['RF', f'{total_values[3]:.1f}'],
        ['ALU', f'{total_values[4]:.1f}'],
        ['', ''],  # Separator row
        ['Total Energy', f'{total_network_energy_value:.1f} μJ'],
        ['', f'({total_network_energy_value/1000:.2f} mJ)']
    ]
    
    # Create the table
    table = ax3.table(cellText=table_data, 
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.5, 0.4])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator row
    table[(6, 0)].set_facecolor('#EEEEEE')
    table[(6, 1)].set_facecolor('#EEEEEE')
    
    # Style total rows
    for i in range(2):
        table[(7, i)].set_facecolor('#FFE5B4')
        table[(7, i)].set_text_props(weight='bold')
        table[(8, i)].set_facecolor('#FFE5B4')
        table[(8, i)].set_text_props(weight='bold')
    
    # Set title for table
    ax3.set_title('Total AlexNet Energy Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/jinhyong/SCALE-Sim-2/alexnet_energy_values.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Print Total AlexNet Energy Summary Table
    total_network_energy_value = sum(total_network_energy.values()) / 1e6  # Convert to μJ
    print("\n" + "="*50)
    print("TOTAL ALEXNET ENERGY SUMMARY")
    print("="*50)
    print(f"DRAM:        {total_values[0]:>10.1f} μJ")
    print(f"SRAM:        {total_values[1]:>10.1f} μJ") 
    print(f"PE_Transfer: {total_values[2]:>10.1f} μJ")
    print(f"RF:          {total_values[3]:>10.1f} μJ")
    print(f"ALU:         {total_values[4]:>10.1f} μJ")
    print("-" * 50)
    print(f"Total Energy: {total_network_energy_value:>9.1f} μJ ({total_network_energy_value/1000:.2f} mJ)")
    print("="*50)
    
    # Create additional detailed numerical table
    print("\n" + "="*80)
    print("DETAILED ENERGY VALUES BY LAYER AND COMPONENT")
    print("="*80)
    
    # Print layer-wise detailed breakdown
    print(f"\n{'Layer':<8} {'DRAM':<12} {'SRAM':<12} {'PE_Transfer':<12} {'RF':<12} {'ALU':<12} {'Total':<12}")
    print("-" * 80)
    
    for layer in layers:
        layer_total = sum(layer_data[layer].values()) / 1e6  # Convert to μJ
        print(f"{layer:<8} ", end="")
        for component in components:
            value = layer_data[layer][component] / 1e6  # Convert to μJ
            print(f"{value:<12.2f} ", end="")
        print(f"{layer_total:<12.2f}")
    
    # Print total row
    print("-" * 80)
    print(f"{'TOTAL':<8} ", end="")
    network_total = 0
    for component in components:
        value = total_network_energy[component] / 1e6  # Convert to μJ
        network_total += value
        print(f"{value:<12.2f} ", end="")
    print(f"{network_total:<12.2f}")
    print("="*80)

def plot_layer_energy_breakdown():
    """Plot energy breakdown for each layer"""
    # Define file paths
    base_path = "./outputs/eyeriss"
    layer_files = {
        'Conv1': f'{base_path}/alexnet_Conv1_energy.csv',
        'Conv2': f'{base_path}/alexnet_Conv2_energy.csv', 
        'Conv3': f'{base_path}/alexnet_Conv3_energy.csv',
        'Conv4': f'{base_path}/alexnet_Conv4_energy.csv',
        'Conv5': f'{base_path}/alexnet_Conv5_energy.csv'
    }
    
    # Read and process data for each layer
    layer_data = {}
    total_network_energy = {'DRAM': 0, 'SRAM': 0, 'PE_Transfer': 0, 'RF': 0, 'ALU': 0}
    
    for layer_name, filepath in layer_files.items():
        if os.path.exists(filepath):
            df = read_energy_csv(filepath)
            grouped = group_components(df)
            layer_data[layer_name] = grouped
            
            # Print detailed energy breakdown for each layer
            print_layer_energy_breakdown(layer_name, df)
            
            # Add to total network energy
            for component, energy in grouped.items():
                total_network_energy[component] += energy
        else:
            print(f"Warning: {filepath} not found")
    
    # Create subplot for layer-wise energy breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Layer-wise energy breakdown (percentage)
    layers = list(layer_data.keys())
    components = ['DRAM', 'SRAM', 'PE_Transfer', 'RF', 'ALU']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Calculate percentages for each layer
    layer_percentages = {}
    for layer in layers:
        total_layer_energy = sum(layer_data[layer].values())
        layer_percentages[layer] = {
            comp: (layer_data[layer][comp] / total_layer_energy) * 100 
            for comp in components
        }
    
    # Create stacked bar chart for layers
    bottom = np.zeros(len(layers))
    for i, component in enumerate(components):
        values = [layer_percentages[layer][component] for layer in layers]
        ax1.bar(layers, values, bottom=bottom, label=component, color=colors[i])
        bottom += values
    
    ax1.set_title('Energy Breakdown by Layer (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy Percentage (%)', fontsize=12)
    ax1.set_xlabel('AlexNet Layers', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for i, layer in enumerate(layers):
        y_pos = 0
        for component in components:
            percentage = layer_percentages[layer][component]
            if percentage > 5:  # Only show labels for segments > 5%
                ax1.text(i, y_pos + percentage/2, f'{percentage:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=9)
            y_pos += percentage
    
    # Plot 2: Total network energy breakdown
    total_energy = sum(total_network_energy.values())
    network_percentages = {
        comp: (energy / total_energy) * 100 
        for comp, energy in total_network_energy.items()
    }
    
    components_list = list(network_percentages.keys())
    percentages_list = list(network_percentages.values())
    
    bars = ax2.bar(components_list, percentages_list, color=colors)
    ax2.set_title('Total AlexNet Energy Breakdown (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy Percentage (%)', fontsize=12)
    ax2.set_xlabel('Components', fontsize=12)
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, percentages_list):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/jinhyong/SCALE-Sim-2/alexnet_energy_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Create additional energy value plots
    plot_energy_values(layer_data, total_network_energy)
    
    # Print summary statistics
    print("="*60)
    print("ALEXNET ENERGY CONSUMPTION ANALYSIS")
    print("="*60)
    
    print("\n1. Layer-wise Energy Consumption (nJ):")
    for layer in layers:
        total_layer_energy = sum(layer_data[layer].values()) / 1e12  # Convert to nJ
        print(f"   {layer}: {total_layer_energy:.2f} nJ")
    
    print(f"\n2. Total Network Energy: {total_energy/1e12:.2f} nJ")
    
    print("\n3. Component Breakdown (% of total network):")
    for component, percentage in network_percentages.items():
        energy_nj = total_network_energy[component] / 1e12
        print(f"   {component}: {percentage:.1f}% ({energy_nj:.2f} nJ)")
    
    print("\n4. Energy Distribution Analysis:")
    # Find most energy-consuming layer
    layer_energies = {layer: sum(data.values()) for layer, data in layer_data.items()}
    max_layer = max(layer_energies, key=layer_energies.get)
    max_energy_pct = (layer_energies[max_layer] / total_energy) * 100
    print(f"   Most energy-consuming layer: {max_layer} ({max_energy_pct:.1f}% of total)")
    
    # Find dominant component
    max_component = max(network_percentages, key=network_percentages.get)
    print(f"   Dominant energy component: {max_component} ({network_percentages[max_component]:.1f}% of total)")

def main():
    """Main function"""
    print("Generating AlexNet Energy Consumption Visualization...")
    plot_layer_energy_breakdown()
    print("Visualization saved as 'alexnet_energy_breakdown.png'")

if __name__ == "__main__":
    main()