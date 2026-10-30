$path = "C:\Users\Praveen\Desktop\AI LOCAL\START.bat"
$bytes = [System.IO.File]::ReadAllBytes($path)
# Remove UTF-8 BOM (EF BB BF) if present
if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    $bytes = $bytes[3..($bytes.Length-1)]
    Write-Host "BOM removed."
} else {
    Write-Host "No BOM found."
}
[System.IO.File]::WriteAllBytes($path, $bytes)
Write-Host "Done. START.bat saved without BOM."
