#!/bin/bash
# Direct fix for the FeatherDataHandler on the server
# Run this script on your server as root

echo "Applying Feather date column fix to server's freqtrade installation..."

# Fix the FeatherDataHandler file directly
cat > /tmp/fix_feather.py << 'EOF'
import sys

file_path = "/home/freqtrade/freqtrade/data/history/datahandlers/featherdatahandler.py"

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_code = """            pairdata = read_compressed_ipc_to_pandas(filename)
            
            # Ensure column names match expected format
            if len(pairdata.columns) == len(self._columns):
                pairdata.columns = self._columns"""
    
    new_code = """            pairdata = read_compressed_ipc_to_pandas(filename)
            
            # Only reassign columns if they don't match expected format
            # Check if columns are already correct before reassigning
            if list(pairdata.columns) != list(self._columns):
                # Only reassign if we have the right number of columns
                if len(pairdata.columns) == len(self._columns):
                    pairdata.columns = self._columns"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w') as f:
            f.write(content)
        print("✓ Fix applied successfully!")
    elif "Only reassign columns if they don't match expected format" in content:
        print("✓ Fix already applied!")
    else:
        # Try alternative pattern without extra whitespace
        old_code2 = """pairdata = read_compressed_ipc_to_pandas(filename)
            
            # Ensure column names match expected format
            if len(pairdata.columns) == len(self._columns):
                pairdata.columns = self._columns"""
        
        if old_code2 in content:
            content = content.replace(old_code2, new_code)
            with open(file_path, 'w') as f:
                f.write(content)
            print("✓ Fix applied successfully!")
        else:
            print("⚠ Could not find expected pattern. Manual fix required.")
            print("Please edit: " + file_path)
            sys.exit(1)
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

# Run the Python fix script
python3 /tmp/fix_feather.py

# Clean up
rm /tmp/fix_feather.py

echo "Fix applied! Try running backtesting again."
