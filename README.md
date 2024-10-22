
# Weighted Box Fusion with COCO CSV Files

## Usage

1. **Install Required Packages**  
   Install the necessary packages using the following command:
   ```bash
   pip install pandas ensemble-boxes
   ```

2. **Run the Script**  
   Execute the script with the following command:
   ```bash
   python wbf_script.py -f <file1.csv> <file2.csv> ... --work-dir <output directory> --output-file <output filename>
   ```

3. **Argument Description**  
   - `-f, --files`: List of CSV files containing prediction results (multiple files allowed)  
   - `--work-dir`: Directory to save the output file (default: current directory)  
   - `--output-file`: Name of the output file (default: `output.csv`)

4. **Example Command**
   ```bash
   python wbf_script.py -f pred1.csv pred2.csv --work-dir ./results --output-file final_wbf.csv
   ```

   This command merges the predictions from `pred1.csv` and `pred2.csv` using WBF and saves the result as `final_wbf.csv` in the `./results` directory.
