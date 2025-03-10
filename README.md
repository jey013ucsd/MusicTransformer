# Music Transformer ECE176

## How to run:

1.

 cd "C:\Users\coolj\Desktop\MusicTransformer\datasets\raw_midi\lmd_full_direct"

 attrib -r /s /d *.mid

 Get-ChildItem "C:\Users\coolj\Desktop\MusicTransformer\datasets\raw_midi\lmd_full_direct" -Recurse -File | Select-Object Name, @{Name="Owner";Expression={(Get-Acl $_.FullName).Owner}}

takeown /F "C:\Users\coolj\Desktop\MusicTransformer\datasets\raw_midi\lmd_full_direct" /R /D Y

$folder = "C:\Users\coolj\Desktop\MusicTransformer\datasets\raw_midi\lmd_full_direct"
$acl = Get-Acl $folder
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("$env:UserName", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
$acl.SetAccessRule($rule)
Set-Acl -Path $folder -AclObject $acl

icacls "C:\Users\coolj\Desktop\MusicTransformer\datasets\raw_midi\lmd_full_direct" /grant Everyone:F /T /C
