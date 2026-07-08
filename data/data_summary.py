import os
import glob
from utils.helper import load_config
from data.data_loader import check_and_generate_mock_data, load_tyre_condition_images, load_rul_df

def get_dataset_status(directory, subdirs, total_files):
    if total_files == 0:
        return "EMPTY"
    for subdir in subdirs:
        target_dir = os.path.join(directory, subdir)
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            if any("mock" in f for f in files):
                return "LOADED (MOCK)"
    return "LOADED (REAL)"

def print_dataset_summary():
    """
    Print sample counts and class balance statistics for all 5 datasets.
    """
    config = load_config()
    
    # Ensure directories and mocks are prepared
    check_and_generate_mock_data(config)
    
    raw_dir = config["data"]["raw_dir"]
    print("=" * 60)
    print("           TIRE DATASET INVENTORY & DIAGNOSTICS")
    print("=" * 60)
    print(f"Data root: {os.path.abspath(raw_dir)}\n")
    
    # 1. TyreNet (Mendeley)
    tyrenet_dir = os.path.join(raw_dir, config["data"]["tyrenet"])
    good_tyrenet = len(glob.glob(os.path.join(tyrenet_dir, "**", "good", "*.jpg"), recursive=True))
    defective_tyrenet = len(glob.glob(os.path.join(tyrenet_dir, "**", "defective", "*.jpg"), recursive=True))
    total_tyrenet = good_tyrenet + defective_tyrenet
    status_tyrenet = get_dataset_status(tyrenet_dir, ["good", "defective"], total_tyrenet)
    print("1. TyreNet Dataset:")
    print(f"   - Good: {good_tyrenet} images")
    print(f"   - Defective: {defective_tyrenet} images")
    print(f"   - Total: {total_tyrenet} images")
    print(f"   - Status: {status_tyrenet}\n")
    
    # 2. Digital images of defective and good condition tyres (Tyre Quality)
    tyre_qual_dir = os.path.join(raw_dir, config["data"]["tyre_quality"])
    good_qual = len(glob.glob(os.path.join(tyre_qual_dir, "**", "good", "*.jpg"), recursive=True))
    defective_qual = len(glob.glob(os.path.join(tyre_qual_dir, "**", "defective", "*.jpg"), recursive=True))
    total_qual = good_qual + defective_qual
    status_qual = get_dataset_status(tyre_qual_dir, ["good", "defective"], total_qual)
    print("2. Tyre Quality Classification Dataset:")
    print(f"   - Good: {good_qual} images")
    print(f"   - Defective: {defective_qual} images")
    print(f"   - Total: {total_qual} images")
    print(f"   - Status: {status_qual}\n")
    
    # 3. Tyre Condition Classification (Primary Wear Classifier Dataset)
    img_paths, labels = load_tyre_condition_images(config)
    class_names = ["New", "Serviceable", "Unusable"]
    print("3. Tyre Condition Classification Dataset (Primary Image Set):")
    if len(img_paths) == 0:
        print("   - No images found.")
    else:
        for c_name in class_names:
            c_count = sum(1 for label in labels if label == class_names.index(c_name))
            pct = (c_count / len(img_paths)) * 100
            print(f"   - {c_name}: {c_count} images ({pct:.1f}%)")
        print(f"   - Total: {len(img_paths)} images")
    is_mock_cond = "LOADED (MOCK)" if any("mock" in os.path.basename(f) for f in img_paths) else "LOADED (REAL)"
    print(f"   - Status: {is_mock_cond}\n")
    
    # 4. Tire Texture Image Recognition
    tire_tex_dir = os.path.join(raw_dir, config["data"]["tire_texture"])
    normal_tex = len(glob.glob(os.path.join(tire_tex_dir, "**", "normal", "*.jpg"), recursive=True))
    cracked_tex = len(glob.glob(os.path.join(tire_tex_dir, "**", "cracked", "*.jpg"), recursive=True))
    total_tex = normal_tex + cracked_tex
    status_tex = get_dataset_status(tire_tex_dir, ["normal", "cracked"], total_tex)
    print("4. Tire Texture Dataset:")
    print(f"   - Normal: {normal_tex} images")
    print(f"   - Cracked/Defective: {cracked_tex} images")
    print(f"   - Total: {total_tex} images")
    print(f"   - Status: {status_tex}\n")
    
    # 5. Synthetic Automobile-Tyre RUL Data (Primary Tabular Set)
    try:
        df = load_rul_df(config)
        print("5. Synthetic Automobile-Tyre RUL Dataset (Primary Tabular Set):")
        print(f"   - Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("   - Feature Statistics:")
        print(f"     * kilometers_driven (km)   : mean={df['kilometers_driven(km)'].mean():.1f}, min={df['kilometers_driven(km)'].min():.1f}, max={df['kilometers_driven(km)'].max():.1f}")
        print(f"     * expected_tyre_life (km)  : mean={df['expected_tyre_life(km)'].mean():.1f}, min={df['expected_tyre_life(km)'].min():.1f}, max={df['expected_tyre_life(km)'].max():.1f}")
        print(f"     * current_tread_depth (mm) : mean={df['current_tread_depth(mm)'].mean():.2f}, min={df['current_tread_depth(mm)'].min():.2f}, max={df['current_tread_depth(mm)'].max():.2f}")
        print(f"     * remaining_useful_life (km): mean={df['remaining_useful_life(km)'].mean():.1f}, min={df['remaining_useful_life(km)'].min():.1f}, max={df['remaining_useful_life(km)'].max():.1f}")
        
        rul_subdir = os.path.join(raw_dir, config["data"]["synthetic_rul"])
        files = os.listdir(rul_subdir)
        is_mock_rul = "LOADED (MOCK)" if any("mock" in f or "synthetic" in f for f in files) else "LOADED (REAL)"
        print(f"   - Status: {is_mock_rul}\n")
    except Exception as e:
        print(f"5. Synthetic Automobile-Tyre RUL Dataset Error: {str(e)}\n")
        
    print("=" * 60)

if __name__ == "__main__":
    print_dataset_summary()
