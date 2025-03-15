# MIDI Transformer ECE176

## How to Use:

1. ### Get datasets 
    ```
    # Downloads and extracts Lakh Midi Dataset
    python get_datasets.py

    ```

2. ### Tokenize dataset
```
# Allow access to files
cd "datasets\raw_midi\lmd_full_direct"

attrib -r /s /d *.mid

Get-ChildItem "datasets\raw_midi\lmd_full_direct" -Recurse -File | Select-Object Name, @{Name="Owner";Expression={(Get-Acl $_.FullName).Owner}}

takeown /F "datasets\raw_midi\lmd_full_direct" /R /D Y

$folder = "datasets\raw_midi\lmd_full_direct"
$acl = Get-Acl $folder
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("$env:UserName", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
$acl.SetAccessRule($rule)
Set-Acl -Path $folder -AclObject $acl

icacls "datasets\raw_midi\lmd_full_direct" /grant Everyone:F /T /C
cd ../../..

# Clean, tokenize, and split dataset
python data_processing/preprocess_data.py # Make sure to set sample and batch size

```

3. ### Train model
```
# Train model, make sure to set desired hyperparameters
python train.py
```

4. ### Test Trained Model
```
# Run inference, make sure to set seed prompt in inference.py
python inference.py
```