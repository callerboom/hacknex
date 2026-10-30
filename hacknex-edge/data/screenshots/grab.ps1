
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
 = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
 = New-Object System.Drawing.Bitmap .Width, .Height
 = [System.Drawing.Graphics]::FromImage()
.CopyFromScreen(.Location, [System.Drawing.Point]::Empty, .Size)
.Save('data/screenshots/test_ps1.png')
.Dispose()
.Dispose()
