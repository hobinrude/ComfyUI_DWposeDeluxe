import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import math
from . import logger

def generate_with_progress(log_path):
    if not os.path.exists(log_path):
        return

    from .progress import progress

    try:
        try:
            rel_path = os.path.basename(log_path)
            if "output" in log_path:
                 rel_path = f"output/{rel_path}"
        except:
            rel_path = log_path
        logger.info(f"Generating memory plot from {rel_path}...")
        
        with open(log_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        pbar = progress(total_lines, label="Plot")
        create_memplot(log_path, progress_callback=pbar.step)
        pbar.finish()
        
    except Exception as e:
        logger.error(f"Failed to create memory plot: {e}")

def create_memplot(log_path, progress_callback=None):
    if not os.path.exists(log_path):
        return

    steps = []
    ram_data = []
    vram_data = []
    
    sections = [] 
    
    metadata = {}
    current_label = "Init"
    ram_total = 0.0
    vram_total = 0.0
    ram_total_from_log = 0.0
    vram_total_from_log = 0.0
    
    sections.append({'start': 0, 'label': current_label})

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if progress_callback:
                progress_callback()

            line = line.strip()
            if not line:
                continue

            if line.startswith("# Metadata:"):
                try:
                    json_str = line.split(":", 1)[1].strip()
                    metadata = json.loads(json_str)
                except:
                    pass
            
            elif line.startswith("# [Loop:"):
                try:
                    content = line[8:-1]
                    parts = [p.strip() for p in content.split(",")]
                    
                    new_label = parts[0]
                    
                    for p in parts:
                        if "RAM_Total" in p: 
                            val = float(p.split(":")[1])
                            if val > 0: ram_total = val
                            if val > 0 and ram_total_from_log == 0:
                                ram_total_from_log = val
                        if "VRAM_Total" in p: 
                            val = float(p.split(":")[1])
                            if val > 0: vram_total = val
                            if val > 0 and vram_total_from_log == 0:
                                vram_total_from_log = val

                    if len(steps) > sections[-1]['start']:
                        current_label = new_label
                        sections.append({'start': len(steps), 'label': current_label})
                    else:
                        sections[-1]['label'] = new_label
                        current_label = new_label

                except Exception as e:
                    print(f"Error parsing loop header: {e}")
            
            else:
                try:
                    parts = line.split(",")
                    steps.append(len(steps) + 1)
                    ram_data.append(float(parts[1]))
                    vram_data.append(float(parts[2]))
                except:
                    pass

    except Exception as e:
        print(f"Failed to parse log file: {e}")
        return

    if not steps:
        return

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    title_str = "Memory Log"
    if metadata:
        provider = metadata.get('provider', '?')
        batch = metadata.get('batch_size', '?')
        w = metadata.get('width', '?')
        h = metadata.get('height', '?')
        poses = metadata.get('poses_to_detect', '?')
        title_str = f"Memory Log | {provider} | Batch: {batch} | {w}x{h} | Poses: {poses}"
    
    plt.title(title_str, color='white', fontsize=10, pad=20)

    start_ram = ram_data[0]
    delta_ram = [max(0, x - start_ram) for x in ram_data]

    color_ram = '#FF0000'        # Red
    color_ram_delta = '#800000'  # Dark-red
    color_vram = '#0064C8'       # Blue
    color_vram_delta = '#003264' # Dark-blue

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('RAM (GB)', color='white')
    
    line1, = ax1.plot(steps, ram_data, color=color_ram, linewidth=4.0, label='System RAM')
    
    ax1.tick_params(axis='y', labelcolor=color_ram)
    
    max_ram_val = ram_total_from_log if ram_total_from_log > 0 else (max(ram_data) if ram_data else 10)
    ylim_ram_calculated = math.ceil(max_ram_val)
    if ylim_ram_calculated == max_ram_val: ylim_ram_calculated += 1
    ylim_ram_buffered = ylim_ram_calculated * 1.1
    ax1.set_ylim(0, ylim_ram_buffered)
    
    if ram_total_from_log > 0:
        ax1.axhline(y=ram_total_from_log, color=color_ram, linestyle='--', alpha=0.5, linewidth=1.0)
        ax1.text(steps[0], ram_total_from_log, f' Total: {ram_total_from_log:.1f}GB', color=color_ram, va='bottom', fontsize=8)

    ax1_delta = ax1.twinx()
    ax1_delta.set_axis_off() 
    
    max_dram = max(delta_ram) if delta_ram else 0
    if max_dram > 0: ax1_delta.set_ylim(0, max_dram * 1.1)

    line1_d, = ax1_delta.plot(steps, delta_ram, color=color_ram_delta, linewidth=2.0, linestyle=':', label='Process RAM Delta')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('VRAM (GB)', color='white')
    ax2.tick_params(axis='y', labelcolor=color_vram)

    max_vram_val = vram_total_from_log if vram_total_from_log > 0 else max(vram_data) if vram_data else 0
    ylim_vram_calculated = math.ceil(max_vram_val)
    if ylim_vram_calculated == max_vram_val: ylim_vram_calculated += 1
    if ylim_vram_calculated == 0: ylim_vram_calculated = 1
    
    ylim_vram_buffered = ylim_vram_calculated * 1.1
    ax2.set_ylim(0, ylim_vram_buffered)

    lines_for_legend = [line1, line1_d]
    names_for_legend = ['System RAM', 'Process RAM Delta']

    if vram_total == 0 and (not vram_data or max(vram_data) == 0):
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0])
    else:
        ax2.set_ylim(0, ylim_vram_buffered)
        
        start_vram = vram_data[0] if vram_data else 0
        delta_vram = [max(0, x - start_vram) for x in vram_data]

        line2, = ax2.plot(steps, vram_data, color=color_vram, linewidth=4.0, label='System VRAM')
        
        if vram_total_from_log > 0:
            ax2.axhline(y=vram_total_from_log, color=color_vram, linestyle='--', alpha=0.5, linewidth=1.0)
            ax2.text(steps[-1] if steps else 0, vram_total_from_log, f'Total: {vram_total_from_log:.1f}GB ', color=color_vram, va='bottom', ha='right', fontsize=8)

        ax2_delta = ax1.twinx()
        ax2_delta.set_axis_off()
        
        max_dvram = max(delta_vram) if delta_vram else 0
        if max_dvram > 0: ax2_delta.set_ylim(0, max_dvram * 1.1)

        line2_d, = ax2_delta.plot(steps, delta_vram, color=color_vram_delta, linewidth=2.0, linestyle=':', label='Process VRAM Delta')
        
        lines_for_legend.extend([line2, line2_d])
        names_for_legend.extend(['System VRAM', 'Process VRAM Delta'])

    ax1.legend(lines_for_legend, names_for_legend, loc='lower right', facecolor='black', framealpha=0.7)

    custom_ticks = []
    custom_labels = []
    
    y_text_pos = ax1.get_ylim()[1] * 0.95

    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.3)
    if steps:
        ax1.axvline(x=steps[-1], color='white', linestyle='--', alpha=0.3)

    for i, sec in enumerate(sections):
        start_idx = sec['start']
        end_idx = sections[i+1]['start'] if i+1 < len(sections) else len(steps)
        
        if end_idx > start_idx:
            duration = end_idx - start_idx
            
            if start_idx > 0:
                ax1.axvline(x=start_idx, color='white', linestyle='--', alpha=0.3)
            
            mid_point = (start_idx + end_idx) / 2
            ax1.text(mid_point, y_text_pos, sec['label'], 
                     color='white', ha='center', va='top', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

            step_size = max(1, duration // 8) 
            local_tick_vals = list(range(0, duration + 1, step_size))
            if local_tick_vals[-1] != duration:
                local_tick_vals.append(duration)
            
            if len(local_tick_vals) > 12:
                step_size = max(1, duration // 5)
                local_tick_vals = list(range(0, duration + 1, step_size))
                if local_tick_vals[-1] != duration: local_tick_vals.append(duration)

            for lt in local_tick_vals:
                global_pos = start_idx + lt
                
                if lt == 0:
                    global_pos += (duration * 0.03) 
                elif lt == duration:
                    global_pos -= (duration * 0.03)
                
                custom_ticks.append(global_pos)
                custom_labels.append(str(lt))

    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels(custom_labels, fontsize=8)

    png_path = log_path.replace('.log', '.png')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    
    try:
        rel_path = os.path.basename(png_path)
        if "output" in png_path:
             rel_path = f"output/{rel_path}"
    except:
        rel_path = png_path
    print()
    logger.info(f"Memory plot saved to: {rel_path}")