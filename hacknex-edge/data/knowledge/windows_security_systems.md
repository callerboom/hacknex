# Windows Security, Sysmon Event IDs, Splunk Queries, and System Administration Guide
HackNEX 2026 — EDRA Edge AI Brain Ground-Truth Knowledge Base

---

## 1. Windows Security Event Log IDs (Authentication & Account Activity)

In Windows Security Event Log (and Splunk SIEM), user logons and authentication events are identified by the following official Event IDs:

- **Event ID 4625 (Failed Logon)**:
  - Meaning: An account failed to log on.
  - Generated on: The system where the logon attempt occurred (Domain Controller for domain accounts, local machine for local accounts).
  - Common Failure Reason Status Codes:
    - `0xC000006A`: User name is correct, but password is wrong.
    - `0xC0000064`: User name does not exist.
    - `0xC000006E`: Account restriction (e.g. workstation restriction).
    - `0xC000006F`: User logon outside authorized hours.
    - `0xC0000071`: Expired password.
    - `0xC0000072`: Account is currently disabled.
    - `0xC0000234`: User account has been locked out due to excessive failed attempts.
  - Important Logon Types for Event 4625:
    - Logon Type 2: Interactive (Logon at local keyboard and screen).
    - Logon Type 3: Network (Connection over SMB, network share, or IIS).
    - Logon Type 7: Unlock (Workstation unlock attempt).
    - Logon Type 10: RemoteInteractive (Remote Desktop Protocol / RDP logon).

- **Event ID 4624 (Successful Logon)**:
  - Meaning: An account was successfully authenticated and logged on.
  - Captures: Account Name, Domain, Logon Type, Source IP Address (Source Network Address), and Process Information.

- **Event ID 4634 & 4647 (Logoff)**:
  - Event ID 4634: An account was logged off.
  - Event ID 4647: User initiated logoff.

- **Event ID 4740 (Account Lockout)**:
  - Meaning: A user account was locked out after reaching the account lockout threshold.

- **Event ID 4720 (Account Created)**:
  - Meaning: A user account was created.

- **Event ID 4726 (Account Deleted)**:
  - Meaning: A user account was deleted.

- **Event ID 4728 / 4732 / 4756 (Security Group Modification)**:
  - Meaning: A member was added to a security group (e.g., Domain Admins or Local Administrators).

- **Event ID 4688 (Process Creation - Windows Native)**:
  - Meaning: A new process has been created (requires Process Creation auditing enabled in GPO).

- **Event ID 7045 (New Service Installed - System Log)**:
  - Meaning: A new service was installed in the Windows system (indicates persistence or malware).

- **Event ID 1102 (Audit Log Cleared)**:
  - Meaning: The Windows Security audit log was cleared by a user or administrator.

---

## 2. Microsoft Sysmon (System Monitor) Event IDs Reference

Microsoft Sysinternals Sysmon is an advanced system monitoring agent.
**CRITICAL NOTE**: Sysmon does NOT log user logon or authentication events directly!
Logon sessions and failed logons are captured exclusively by the Windows Security Event Log as **Event ID 4625** (failed logon) and **Event ID 4624** (successful logon).

Sysmon monitors low-level process, network, and disk events:
- **Sysmon Event ID 1 (Process Creation)**:
  - Captures: Process command line, executable path (Image), ParentProcessId, ParentImage, hashes (MD5, SHA256), and user account.
- **Sysmon Event ID 2 (Process Changed File Creation Time)**:
  - Detects timestomping (attackers modifying file timestamps to blend with system files).
- **Sysmon Event ID 3 (Network Connection)**:
  - Logs TCP/UDP network connections, destination IP, destination port, hostname, and the executable that initiated it.
- **Sysmon Event ID 4 (Sysmon Service State Changed)**:
  - Logs when the Sysmon service is started or stopped.
- **Sysmon Event ID 5 (Process Terminated)**:
  - Logs when a process ends.
- **Sysmon Event ID 6 (Driver Loaded)**:
  - Logs drivers loaded by the Windows kernel.
- **Sysmon Event ID 7 (Image Loaded - DLL)**:
  - Logs when a module or DLL is loaded by a process.
- **Sysmon Event ID 8 (CreateRemoteThread)**:
  - Detects process injection (e.g., thread injection into svchost or explorer).
- **Sysmon Event ID 9 (RawAccessRead)**:
  - Detects raw disk reads (e.g., direct VSS / volume access).
- **Sysmon Event ID 10 (ProcessAccess)**:
  - Logs when a process opens another process handle (essential for detecting LSASS dumping by Mimikatz).
- **Sysmon Event ID 11 (FileCreate)**:
  - Logs when a file is created or written.
- **Sysmon Event ID 12, 13, 14 (Registry Events)**:
  - Event ID 12: Registry object created or deleted.
  - Event ID 13: Registry value set (autorun persistence).
  - Event ID 14: Registry key renamed.
- **Sysmon Event ID 15 (FileCreateStreamHash)**:
  - Detects Alternate Data Streams (ADS).
- **Sysmon Event ID 17 & 18 (Pipe Created / Connected)**:
  - Used for detecting Cobalt Strike and SMB named pipes.
- **Sysmon Event ID 19, 20, 21 (WMI Events)**:
  - Detects WMI Event Filters and Consumers used for fileless persistence.
- **Sysmon Event ID 22 (DNSEvent)**:
  - Logs every DNS query made by applications (critical for C2 domain detection).
- **Sysmon Event ID 23 (FileDelete)**:
  - File deleted and archived.
- **Sysmon Event ID 25 (Process Tampering)**:
  - Detects process hollowing and herpaderping.

---

## 3. Splunk Queries for Windows and Sysmon Analysis

When searching Windows and Sysmon logs in Splunk:

- **Failed Logons in Splunk**:
  ```spl
  index=windows sourcetype=WinEventLog:Security (EventCode=4625 OR EventID=4625)
  | stats count by Account_Name, Source_Network_Address, Failure_Reason
  | sort -count
  ```
- **Brute Force Detection in Splunk**:
  ```spl
  index=windows sourcetype=WinEventLog:Security EventCode=4625
  | bin _time span=5m
  | stats count by Account_Name, Source_Network_Address, _time
  | where count > 5
  ```
- **Successful RDP Logons in Splunk**:
  ```spl
  index=windows sourcetype=WinEventLog:Security EventCode=4624 Logon_Type=10
  | table _time, Account_Name, Source_Network_Address, Workstation_Name
  ```
- **Sysmon Process Execution in Splunk**:
  ```spl
  index=windows sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=1
  | table _time, ComputerName, User, Image, CommandLine, ParentImage
  ```
- **Sysmon Suspicious Network Connections in Splunk**:
  ```spl
  index=windows sourcetype="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3 DestinationPort IN (4444, 1337, 8080)
  | table _time, Image, DestinationIp, DestinationPort
  ```

---

## 4. Windows Problem Troubleshooting & Command Reference

Common Windows system issues, commands, and fixes:

- **Corrupt System Files**:
  - Run Command Prompt as Administrator: `sfc /scannow`
  - If SFC fails or reports unrepairable files: `DISM /Online /Cleanup-Image /RestoreHealth`
  - Restart the computer after DISM completes.

- **Network & DNS Issues**:
  - Clear local DNS resolver cache: `ipconfig /flushdns`
  - Release and renew DHCP IP address: `ipconfig /release` followed by `ipconfig /renew`
  - Test network connectivity to a specific port (PowerShell): `Test-NetConnection -ComputerName google.com -Port 443`
  - View all active network connections and owning PID: `netstat -ano | findstr LISTENING`

- **Port in Use & Process Killing**:
  - Find PID using port 8080: `netstat -ano | findstr :8080`
  - Find process name for PID 1234: `tasklist | findstr 1234`
  - Force kill process by PID: `taskkill /F /PID 1234`
  - In PowerShell: `Stop-Process -Id 1234 -Force`

- **Windows Service Control**:
  - Check service status: `Get-Service -Name "wuauserv"`
  - Restart a service: `Restart-Service -Name "Spooler" -Force`
  - In CMD: `net stop <service_name>` and `net start <service_name>`

- **Blue Screen of Death (BSOD)**:
  - Minidump location: `C:\Windows\Minidump`
  - Common codes:
    - `0x0000003B (SYSTEM_SERVICE_EXCEPTION)`: Buggy driver or corrupt system files.
    - `0x000000D1 (DRIVER_IRQL_NOT_LESS_OR_EQUAL)`: Network or graphics driver conflict.
    - `0x0000001A (MEMORY_MANAGEMENT)`: Faulty RAM hardware or memory driver issue.
  - Windows Memory Diagnostic: Run `mdsched.exe` and reboot to test RAM.

- **Disk & Storage Errors**:
  - Check disk for errors on drive C: `chkdsk C: /f /r`
  - Open Disk Management: `diskmgmt.msc`
  - Disk Part utility: `diskpart`

- **Windows Event Viewer Navigation**:
  - Launch Event Viewer: Press `Win + R`, type `eventvwr.msc`, press Enter.
  - View Security log events in PowerShell:
    `Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4625} -MaxEvents 5`
