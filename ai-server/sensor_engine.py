import csv
import os

# Path to the feature CSV file relative to ai-server
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "features", "ai_input_feature.csv")

def evaluate_sensor_rules(csv_path: str = CSV_PATH):
    """
    Reads ai_input_feature.csv, checks quality_flag, computes baseline_change,
    and returns the latest matched AI Action and GUI Trigger.
    """
    if not os.path.exists(csv_path):
        # Fallback if CSV is located in docs or handover folder
        csv_path = os.path.join(os.path.dirname(__file__), "..", "docs", "AI_Group_Handover", "ai_input_feature.csv")
    
    if not os.path.exists(csv_path):
        return {
            "status": "file_not_found",
            "stress_level": "NORMAL",
            "gui_trigger": "DEFAULT_UI",
            "ai_action": "NORMAL_DIALOGUE"
        }

    latest_valid_row = None

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Check quality_flag (assume 1/True/'good'/'valid' indicates valid data)
                quality_flag = str(row.get("quality_flag", "1")).strip().lower()
                if quality_flag in ["0", "false", "bad", "invalid"]:
                    continue  # Skip low-quality data
                
                latest_valid_row = row

        if not latest_valid_row:
            return {
                "status": "no_valid_quality_data",
                "stress_level": "UNKNOWN",
                "gui_trigger": "DEFAULT_UI",
                "ai_action": "NORMAL_DIALOGUE"
            }

        # 2. Extract values & compute baseline_change if not pre-computed
        eda_val = float(latest_valid_row.get("eda_value", latest_valid_row.get("eda", 0)))
        baseline = float(latest_valid_row.get("baseline", latest_valid_row.get("eda_baseline", eda_val)))
        
        if "baseline_change" in latest_valid_row and latest_valid_row["baseline_change"]:
            baseline_change = float(latest_valid_row["baseline_change"])
        else:
            baseline_change = eda_val - baseline

        # 3. Match against Sensor-AI Rule Matrix
        # (Thresholds can be adjusted according to sensor_ai_rule_matrix.md)
        if baseline_change >= 0.5:
            stress_level = "HIGH_STRESS"
            gui_trigger = "SHOW_STRESS_INDICATOR_RED"
            ai_action = "USE_CALMING_TONE"
        elif baseline_change >= 0.2:
            stress_level = "MODERATE_STRESS"
            gui_trigger = "SHOW_STRESS_INDICATOR_YELLOW"
            ai_action = "SLOW_DIALOGUE_PACE"
        else:
            stress_level = "LOW_STRESS"
            gui_trigger = "SHOW_STRESS_INDICATOR_GREEN"
            ai_action = "STANDARD_DIALOGUE"

        return {
            "status": "success",
            "quality_flag": 1,
            "baseline_change": round(baseline_change, 4),
            "stress_level": stress_level,
            "gui_trigger": gui_trigger,
            "ai_action": ai_action
        }

    except Exception as e:
        return {
            "status": f"error: {str(e)}",
            "stress_level": "NORMAL",
            "gui_trigger": "DEFAULT_UI",
            "ai_action": "NORMAL_DIALOGUE"
        }