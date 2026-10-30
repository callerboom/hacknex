# Windows File System Hierarchy, Directory Structure, and OS Internals Guide
HackNEX 2026 — EDRA Edge AI Brain Ground-Truth Knowledge Base

---

## 1. Windows Directory Structure & Filesystem Layout

On any standard Microsoft Windows installation (Windows 10, Windows 11, Windows Server), the primary system volume is mounted on `C:\`. The directory hierarchy is organized as follows:

### Root Level (`C:\`)
- **`C:\Windows`**: Core operating system directory containing system binaries, DLLs, drivers, and configuration files.
- **`C:\Program Files`**: Default installation path for 64-bit native applications (e.g., Python 64-bit, Git, Google Chrome, VS Code).
- **`C:\Program Files (x86)`**: Default installation directory for 32-bit applications running on 64-bit Windows via the WOW64 subsystem.
- **`C:\ProgramData`**: A hidden system directory used by applications to store configuration data, databases, and application state shared across all users on the computer.
- **`C:\Users`**: Root directory containing individual user profile directories.
- **Hidden Root System Files**:
  - `bootmgr`: Windows Boot Manager file.
  - `pagefile.sys`: Windows virtual memory swap/paging file.
  - `hiberfil.sys`: Hibernate memory snapshot file.
  - `swapfile.sys`: Specialized paging file for Universal Windows Platform (UWP) apps.
  - `System Volume Information`: Restricted system directory storing Volume Shadow Copies (VSS), restore points, and indexing databases.

---

## 2. Windows System Folders Under `C:\Windows`

- **`C:\Windows\System32`**:
  - The single most critical folder in Windows.
  - Contains all native 64-bit Windows core binaries, operating system DLLs, kernel components, and command-line administrative tools.
  - Subdirectories:
    - `C:\Windows\System32\drivers`: Kernel-mode device drivers (`.sys` files).
    - `C:\Windows\System32\drivers\etc\hosts`: Hostname to IP address mapping file.
    - `C:\Windows\System32\config`: Stores the Windows Registry hives (`SYSTEM`, `SOFTWARE`, `SAM`, `SECURITY`, `DEFAULT`).
    - `C:\Windows\System32\wbem`: Windows Management Instrumentation (WMI) binaries and repository.
    - `C:\Windows\System32\Tasks`: Windows Scheduled Tasks definitions.
    - `C:\Windows\System32\GroupPolicy`: Local Group Policy settings.
    - `C:\Windows\System32\winevt\Logs`: Windows Event Log storage (`.evtx` files).

- **`C:\Windows\SysWOW64`**:
  - Name stands for "System Windows-on-Windows 64-bit".
  - Despite the name ending in "64", this folder contains **32-bit** system binaries and DLLs to allow 32-bit applications to execute on 64-bit Windows.

- **`C:\Windows\WinSxS` (Windows Side-by-Side Component Store)**:
  - Stores all component packages, multiple simultaneous versions of DLL assemblies to prevent "DLL Hell", and OS update packages. Most files in System32 are hard links to WinSxS.

- **`C:\Windows\Temp`**:
  - Operating system-level temporary directory.

- **`C:\Windows\Logs` & `C:\Windows\Panther`**:
  - Operating system setup, CBS (Component Based Servicing), and installation logs.

---

## 3. User Profile Structure (`C:\Users\<Username>`)

Every Windows user account has a personal profile directory containing:
- **`Desktop`**: Files and shortcuts displayed on the user's desktop screen.
- **`Documents`**: Default directory for personal documents and user-created files.
- **`Downloads`**: Default target for web downloads and downloaded packages.
- **`Pictures`, `Music`, `Videos`**: Default user media directories.
- **`NTUSER.DAT`**: Hidden file storing the user's specific Registry hive (`HKEY_CURRENT_USER`).
- **`AppData`** (Hidden folder containing user application data):
  - **`AppData\Roaming`** (`%APPDATA%`): Stores configuration and data designed to roam with the user profile across a network domain (e.g., `%APPDATA%\npm`, `%APPDATA%\Code`, `%APPDATA%\Mozilla`).
  - **`AppData\Local`** (`%LOCALAPPDATA%`): Machine-specific data, caches, app installations (e.g., Chrome, Discord, Python user packages, Edge cache).
  - **`AppData\Local\Temp`** (`%TEMP%`): Per-user temporary files directory.
  - **`AppData\LocalLow`**: Low-integrity application data for sandboxed processes (like browsers running in protected mode).

---

## 4. Core Windows System Executables & Services

The fundamental Windows architecture relies on key system processes:

| Process Name | Full Binary Path | Role and Description |
| :--- | :--- | :--- |
| **System** (PID 4) | In-kernel | Kernel worker threads and NT operating system kernel (`ntoskrnl.exe`). |
| **smss.exe** | `C:\Windows\System32\smss.exe` | Session Manager Subsystem. The first user-mode process started by the kernel; launches winlogon and csrss. |
| **csrss.exe** | `C:\Windows\System32\csrss.exe` | Client/Server Runtime Subsystem. Manages console windows and GUI process threading. |
| **wininit.exe** | `C:\Windows\System32\wininit.exe` | Windows Initialization Process. Launches the Service Control Manager (`services.exe`), LSASS, and LSA. |
| **services.exe**| `C:\Windows\System32\services.exe`| Service Control Manager (SCM). Starts, stops, and manages background Windows services and drivers. |
| **lsass.exe** | `C:\Windows\System32\lsass.exe` | Local Security Authority Subsystem Service. Verifies user logins, enforces security policies, creates security access tokens. |
| **svchost.exe** | `C:\Windows\System32\svchost.exe` | Service Host. Generic host process for services executing from dynamic-link libraries (DLLs). |
| **explorer.exe**| `C:\Windows\explorer.exe` | Windows Shell. Provides Desktop, Taskbar, Start Menu, Notification Area, and File Explorer. |
| **conhost.exe** | `C:\Windows\System32\conhost.exe`| Console Window Host. Provides terminal emulation and window management for `cmd.exe` and `powershell.exe`. |
| **taskmgr.exe** | `C:\Windows\System32\taskmgr.exe`| Task Manager. Monitors system performance, CPU/RAM/Disk metrics, and process management. |
| **regedit.exe** | `C:\Windows\regedit.exe` | Registry Editor GUI. |
| **cmd.exe** | `C:\Windows\System32\cmd.exe` | Classic Windows Command Prompt interpreter. |
| **powershell.exe**| `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` | Windows PowerShell scripting and command environment. |

---

## 5. Windows Management Consoles (.msc files)

Windows provides Microsoft Management Consoles located in `C:\Windows\System32\`:
- `services.msc`: Manage background Windows services (start, stop, set startup type).
- `taskschd.msc`: Windows Task Scheduler.
- `eventvwr.msc`: Windows Event Viewer (Application, Security, System logs).
- `devmgmt.msc`: Device Manager (Hardware components, drivers).
- `diskmgmt.msc`: Disk Management (Partitions, drive letters, formatting).
- `compmgmt.msc`: Computer Management (Combines Event Viewer, Disk, Services, and Users).
- `perfmon.msc`: Performance Monitor.
- `gpedit.msc`: Local Group Policy Editor.
- `secpol.msc`: Local Security Policy.

---

## 6. Windows Registry Architecture

The Windows Registry is a hierarchical database that stores low-level settings for the OS and applications.
Root Hives:
1. **`HKEY_LOCAL_MACHINE` (`HKLM`)**: Hardware, driver, and system-wide software configurations for all users.
2. **`HKEY_CURRENT_USER` (`HKCU`)**: User-specific settings for the currently logged-in user (points to `HKEY_USERS\<UserSID>`).
3. **`HKEY_CLASSES_ROOT` (`HKCR`)**: File associations and COM object registration.
4. **`HKEY_USERS` (`HKU`)**: Contains profiles for all users loaded on the machine.
5. **`HKEY_CURRENT_CONFIG` (`HKCC`)**: Information gathered during hardware boot.

Common Registry Run Keys (Used by software startup and malware persistence):
- `HKLM\Software\Microsoft\Windows\CurrentVersion\Run`
- `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
- `HKLM\Software\Microsoft\Windows\CurrentVersion\RunOnce`
