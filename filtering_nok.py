import os
import shutil

def move_nok_bottles():
    """
    Verplaatst alle images met NOK label (class_id = 1) naar test/nok folder.
    Checked zowel train als val folders.
    """
    
    dataset_dir = 'test'
    nok_images_dir = os.path.join(dataset_dir, 'nok')
    os.makedirs(nok_images_dir, exist_ok=True)
    
    # Folders om te checken
    splits = ['train', 'val']
    
    total_moved = 0
    
    for split in splits:
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        images_dir = os.path.join(dataset_dir, 'images', split)
        
        if not os.path.exists(labels_dir):
            print(f"Skipping {labels_dir} (doesn't exist)")
            continue
        
        print(f"\nChecking {labels_dir}...")
        
        # Loop door alle label files
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            
            label_path = os.path.join(labels_dir, label_file)
            
            # Lees het label file
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Check of er een NOK label (1) in zit
            has_nok = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id == 1:  # NOK
                        has_nok = True
                        break
            
            # Als NOK gevonden, verplaats de image
            if has_nok:
                # Zoek de corresponderende image
                base_name = os.path.splitext(label_file)[0]
                image_file = base_name + '.jpg'
                image_path = os.path.join(images_dir, image_file)
                
                if os.path.exists(image_path):
                    dest_path = os.path.join(nok_images_dir, image_file)
                    shutil.move(image_path, dest_path)
                    print(f"Moved: {image_file}")
                    total_moved += 1
                else:
                    print(f"Warning: Image not found for {label_file}")
    
    print(f"\n=== DONE ===")
    print(f"Total NOK images moved: {total_moved}")
    print(f"Destination folder: {nok_images_dir}")

if __name__ == '__main__':
    move_nok_bottles()