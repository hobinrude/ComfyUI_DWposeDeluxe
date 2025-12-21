import curses
import json
import sys
import os

# Updated list of body points including feet based on dwpose/util.py and __init__.py
BODY_POINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def get_person_points(person):
    """Flatten all keypoint types into a single list for display."""
    points = []
    
    # Body + Feet Keypoints (0-23)
    body = person.get("pose_keypoints_2d", [])
    for i in range(0, len(body), 3):
        idx = i // 3
        points.append((f"body_{idx:02d}", body[i], body[i+1], body[i+2]))
    
    # Face Keypoints
    face = person.get("face_keypoints_2d", [])
    for i in range(0, len(face), 3):
        idx = i // 3
        points.append((f"face_{idx:02d}", face[i], face[i+1], face[i+2]))
        
    # Left Hand Keypoints
    lhand = person.get("hand_left_keypoints_2d", [])
    for i in range(0, len(lhand), 3):
        idx = i // 3
        points.append((f"Lhnd_{idx:02d}", lhand[i], lhand[i+1], lhand[i+2]))
        
    # Right Hand Keypoints
    rhand = person.get("hand_right_keypoints_2d", [])
    for i in range(0, len(rhand), 3):
        idx = i // 3
        points.append((f"Rhnd_{idx:02d}", rhand[i], rhand[i+1], rhand[i+2]))
        
    return points

def get_point_style(pid, conf_pct):
    """Returns (color_pair, attr) based on thresholds."""
    if pid.startswith("body_"):
        if conf_pct < 30: return 1, curses.A_BOLD # Bright Red
        if conf_pct <= 50: return 2, curses.A_BOLD # Bright Yellow
        return 3, curses.A_NORMAL # Green
    else:
        if conf_pct < 15: return 1, curses.A_BOLD # Bright Red
        if conf_pct <= 30: return 2, curses.A_BOLD # Bright Yellow
        return 3, curses.A_NORMAL # Green

def safe_addch(stdscr, y, x, char, attr=0):
    """Safely adds a character, ignoring errors at window boundaries."""
    try:
        h, w = stdscr.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            stdscr.addch(y, x, char, attr)
    except:
        pass

def safe_addstr(stdscr, y, x, string, attr=0):
    """Safely adds a string, ignoring errors at window boundaries."""
    try:
        h, w = stdscr.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            stdscr.addstr(y, x, string[:w-x], attr)
    except:
        pass

def main(stdscr, data):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    
    curses.curs_set(0)
    stdscr.keypad(True)
    
    current_frame = 0
    total_frames = len(data)
    scroll_pos = 0
    
    while True:
        stdscr.clear()
        term_h, term_w = stdscr.getmaxyx()
        
        if term_h < 5 or term_w < 20:
            safe_addstr(stdscr, 0, 0, "Terminal too small!")
            stdscr.refresh()
            stdscr.getch()
            continue

        # --- Header (Line 0) ---
        header = f" FRAME: {current_frame + 1:04} / {total_frames:04} | Arrows (Shift for +/-10) | 'q' to Quit "
        safe_addstr(stdscr, 0, 0, header.ljust(term_w), curses.A_REVERSE)
        
        frame_data = data[current_frame]
        people = frame_data.get("people", [])
        canvas_w = frame_data.get("canvas_width", 1)
        canvas_h = frame_data.get("canvas_height", 1)

        if not people:
            safe_addstr(stdscr, term_h // 2, (term_w // 2) - 10, "NO POSES DETECTED", curses.color_pair(1) | curses.A_BOLD)
        else:
            points = get_person_points(people[0])
            
            # --- Data Pane (Left Side) ---
            # Layout: [   pid   ] (Label) | X | Y | C
            list_width = 52
            content_h = term_h - 1
            
            for i in range(content_h):
                p_idx = scroll_pos + i
                if p_idx < len(points):
                    pid, x, y, c = points[p_idx]
                    x_i, y_int, c_p = int(round(x)), int(round(y)), int(round(c * 100))
                    
                    label = f"({BODY_POINT_NAMES[p_idx]})" if pid.startswith("body_") and p_idx < len(BODY_POINT_NAMES) else ""
                    id_box = f"[ {pid:7} ]"
                    line = f"{id_box} {label:12} | X:{x_i:5} | Y:{y_int:5} | {c_p:3}%"
                    
                    cp, attr = get_point_style(pid, c_p)
                    if x < 0 or y < 0: attr |= curses.A_UNDERLINE
                    
                    safe_addstr(stdscr, 1 + i, 0, line, curses.color_pair(cp) | attr)

            # --- Visual Canvas (Right Side) ---
            canvas_start_x = list_width + 1
            v_w = term_w - canvas_start_x - 2
            v_h = term_h - 3
            
            if v_w > 5 and v_h > 5:
                stdscr.attron(curses.A_DIM)
                # Horizontal borders
                for x in range(canvas_start_x, canvas_start_x + v_w + 2):
                    safe_addch(stdscr, 1, x, curses.ACS_HLINE)
                    safe_addch(stdscr, v_h + 2, x, curses.ACS_HLINE)
                # Vertical borders
                for y in range(1, v_h + 3):
                    safe_addch(stdscr, y, canvas_start_x, curses.ACS_VLINE)
                    safe_addch(stdscr, y, canvas_start_x + v_w + 1, curses.ACS_VLINE)
                # Corner characters
                safe_addch(stdscr, 1, canvas_start_x, curses.ACS_ULCORNER)
                safe_addch(stdscr, 1, canvas_start_x + v_w + 1, curses.ACS_URCORNER)
                safe_addch(stdscr, v_h + 2, canvas_start_x, curses.ACS_LLCORNER)
                safe_addch(stdscr, v_h + 2, canvas_start_x + v_w + 1, curses.ACS_LRCORNER)
                stdscr.attroff(curses.A_DIM)
                
                # Plot points on ASCII map
                for pid, x, y, c in points:
                    c_p = int(round(c * 100))
                    norm_x = x / canvas_w
                    norm_y = y / canvas_h
                    
                    # Map normalized coords to available TUI grid space
                    plot_x = canvas_start_x + 1 + int(norm_x * (v_w))
                    plot_y = 2 + int(norm_y * (v_h))
                    
                    if canvas_start_x < plot_x <= canvas_start_x + v_w and 1 < plot_y <= 1 + v_h:
                        cp, attr = get_point_style(pid, c_p)
                        char = "O" if pid.startswith("body_") else "."
                        safe_addch(stdscr, plot_y, plot_x, char, curses.color_pair(cp) | attr)

        stdscr.refresh()
        key = stdscr.getch()
        
        if key == ord('q'): break
        elif key == curses.KEY_RIGHT: current_frame = (current_frame + 1) % total_frames; scroll_pos = 0
        elif key == curses.KEY_LEFT: current_frame = (current_frame - 1) % total_frames; scroll_pos = 0
        elif key in [curses.KEY_SRIGHT, 402, 561, ord('.')]: 
            current_frame = min(total_frames - 1, current_frame + 10); scroll_pos = 0
        elif key in [curses.KEY_SLEFT, 393, 546, ord(',')]: 
            current_frame = max(0, current_frame - 10); scroll_pos = 0
        elif key == curses.KEY_DOWN and people and scroll_pos < len(points) - content_h: scroll_pos += 1
        elif key == curses.KEY_UP and scroll_pos > 0: scroll_pos -= 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_kp.py <json_file>")
        sys.exit(1)
    try:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
            if isinstance(data, dict): data = [data]
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
    curses.wrapper(main, data)
