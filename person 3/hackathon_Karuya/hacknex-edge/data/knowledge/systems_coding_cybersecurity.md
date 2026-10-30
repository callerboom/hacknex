# Systems Coding, Architecture, Operating Systems, Networking, and Cyber Security Guide
HackNEX 2026 — EDRA Edge AI Brain Ground-Truth Knowledge Base

---

## 1. Systems Programming & Memory Architecture

- **Process Memory Layout in Virtual Address Space**:
  - **Text Segment (Code)**: Read-only machine instructions executed by the CPU.
  - **Data Segment**: Initialized global and static variables.
  - **BSS Segment (Block Started by Symbol)**: Uninitialized global and static variables (zero-initialized by kernel).
  - **Heap**: Dynamically allocated memory (`malloc`, `free`, `new`, `delete`), grows upward towards higher memory addresses.
  - **Memory Mapping Segment**: Shared libraries, memory-mapped files (`mmap`), dynamically loaded DLLs.
  - **Stack**: Stores function call frames, local variables, return addresses, and saved registers; grows downward toward lower memory addresses.
- **CPU Architecture & Registers (x86_64)**:
  - General Purpose Registers: `RAX` (accumulator/return value), `RBX` (base), `RCX` (counter), `RDX` (data), `RSI` (source index), `RDI` (destination index).
  - Pointer Registers: `RSP` (Stack Pointer, top of stack), `RBP` (Base/Frame Pointer).
  - Instruction Pointer: `RIP` (Points to next instruction to be fetched).
  - Calling Convention (System V AMD64 ABI): First 6 integer/pointer arguments passed in `RDI`, `RSI`, `RDX`, `RCX`, `R8`, `R9`. Return value in `RAX`.
  - Windows x64 Calling Convention: First 4 arguments in `RCX`, `RDX`, `R8`, `R9`. Caller reserves 32 bytes "shadow space" on the stack.

---

## 2. Operating System Internals & Concurrency

- **Virtual Memory & Paging**:
  - Virtual addresses translated to physical addresses by Memory Management Unit (MMU) using multi-level Page Tables.
  - Translation Lookaside Buffer (TLB): Hardware cache of recent virtual-to-physical translations.
  - Page Fault: Interrupt triggered when a page is not present in physical RAM; kernel fetches page from disk (`pagefile.sys`).
- **Threads vs Processes**:
  - Process: An executing program instance with its own independent virtual address space, file handles, and security context.
  - Thread: The basic unit of CPU execution within a process; all threads in a process share the same address space, heap, and open handles, but have their own stack and register context.
- **Synchronization Primitives**:
  - **Mutex (Mutual Exclusion)**: Locks shared resource so only one thread accesses it at a time.
  - **Semaphore**: Integer counter managing access to a pool of shared resources (Counting semaphore) or binary mutex.
  - **Deadlock Conditions (Coffman Conditions)**:
    1. Mutual Exclusion.
    2. Hold and Wait.
    3. No Preemption.
    4. Circular Wait.

---

## 3. Computer Networking & Protocols

- **OSI 7-Layer vs TCP/IP Model**:
  - Layer 7: Application (HTTP, HTTPS, DNS, SSH, FTP, SMTP).
  - Layer 4: Transport (TCP, UDP).
  - Layer 3: Network (IP, ICMP, ARP).
  - Layer 2: Data Link (Ethernet, Wi-Fi, MAC addresses).
  - Layer 1: Physical (Cables, radio waves, electrical signals).
- **TCP (Transmission Control Protocol)**:
  - Connection-oriented, reliable, ordered delivery, flow control, congestion control.
  - **Three-Way Handshake**:
    1. Client -> Server: `SYN` (seq = x).
    2. Server -> Client: `SYN-ACK` (seq = y, ack = x + 1).
    3. Client -> Server: `ACK` (seq = x + 1, ack = y + 1).
  - **Connection Teardown**: Four-way handshake (`FIN`, `ACK`, `FIN`, `ACK`).
- **UDP (User Datagram Protocol)**:
  - Connectionless, unreliable, low latency, no retransmissions (ideal for VoIP, DNS queries, live video streaming, gaming).
- **DNS (Domain Name System)**:
  - Resolves domain names to IP addresses via root servers, TLD servers, and authoritative name servers over UDP/TCP port 53.
- **TLS/SSL Handshake (TLS 1.3)**:
  - ClientHello with supported cipher suites and Diffie-Hellman key share -> ServerHello with chosen cipher and server key share -> encrypted session established in 1 RTT.

---

## 4. Cyber Security, Exploitation & Defense

- **Common Web & Software Vulnerabilities (OWASP Top 10)**:
  - **SQL Injection (SQLi)**: Untrusted input concatenated into SQL queries; defense: Parameterized queries / Prepared Statements.
  - **Cross-Site Scripting (XSS)**: Malicious JavaScript executed in victim browser; types: Stored, Reflected, DOM-based; defense: Content Security Policy (CSP), HTML output encoding.
  - **Cross-Site Request Forgery (CSRF)**: Forcing an authenticated browser to submit unauthorized requests; defense: Anti-CSRF tokens, SameSite cookies.
  - **Server-Side Request Forgery (SSRF)**: Attacker abuses server functionality to access internal network resources or cloud metadata (`http://169.254.169.254`).
- **Binary Exploitation Techniques & Mitigations**:
  - **Buffer Overflow**: Writing past allocated stack buffer to overwrite the saved return address (`RIP`/`EIP`).
  - **Stack Canaries**: Random value placed before return address; compiler checks value before returning (`/GS` on MSVC, `-fstack-protector` on GCC).
  - **ASLR (Address Space Layout Randomization)**: Randomizes memory offsets of stack, heap, and libraries.
  - **DEP / NX (Data Execution Prevention / No-Execute)**: Marks stack and heap as non-executable; prevents executing injected shellcode.
  - **ROP (Return-Oriented Programming)**: Chaining existing executable code snippets ending in `ret` ("gadgets") to bypass DEP/NX.

---

## 5. Windows Security Mechanisms & Threat Hunting

- **Windows Authentication Protocols**:
  - **Kerberos**: Default Active Directory protocol using Tickets (TGT from KDC, Service Tickets TGS), port 88.
  - **NTLM**: Legacy challenge-response authentication protocol based on MD4 password hashes.
- **Common Enterprise Windows Attacks**:
  - **Pass-the-Hash (PtH)**: Lateral movement using captured NTLM hash without cracking the plaintext password.
  - **Kerberoasting**: Requesting TGS service tickets for accounts with Service Principal Names (SPNs) and cracking ticket offline.
  - **LSASS Memory Dumping**: Reading `lsass.exe` memory (e.g., using Mimikatz or ProcDump) to extract cleartext passwords and hashes.
  - **Golden Ticket**: Forged Kerberos TGT encrypted with the domain's `krbtgt` account hash, granting full domain dominance.
- **Windows Blue Team Defenses**:
  - **AMSI (Antimalware Scan Interface)**: Allows applications (PowerShell, VBScript, Excel) to send obfuscated scripts to antivirus before execution.
  - **Credential Guard**: Uses Virtualization-Based Security (VBS) to isolate LSASS secrets in a protected Hyper-V micro-VM (`LsaIso.exe`).
  - **Sysmon & Event Log Auditing**: Event ID 4624 (Logon), Event ID 4625 (Failed Logon), Sysmon ID 1 (Process Create), Sysmon ID 10 (ProcessAccess on LSASS).

---

## 6. Cryptography Principles

- **Symmetric Encryption**:
  - Same key used for encryption and decryption.
  - **AES (Advanced Encryption Standard)**: 128, 192, 256-bit keys. Block size 128 bits.
  - Modes: CBC (requires IV, vulnerable to padding oracle if misused), GCM (Galois/Counter Mode — Authenticated Encryption with Associated Data, fast, AEAD).
- **Asymmetric Encryption & Key Exchange**:
  - **RSA**: Relies on the integer factorization problem of two large prime numbers.
  - **Diffie-Hellman / ECDH**: Allows two parties to negotiate a shared secret over an insecure channel.
  - **Elliptic Curve Cryptography (ECC)**: Provides equivalent security to RSA with significantly smaller key sizes (e.g., Curve25519, secp256k1).
- **Cryptographic Hashing**:
  - One-way, deterministic, avalanche effect, collision resistant.
  - SHA-256, SHA-512, SHA-3.
  - Password Hashing (Salting and Key Derivation Functions): bcrypt, PBKDF2, Argon2id.

---

## 7. Penetration Testing Methodology, Tooling & Ethical Hacking Practice

- **Ethical Hacking Standards & Phases (PTES - Penetration Testing Execution Standard)**:
  1. **Pre-engagement Interactions**: Defining rules of engagement, scope, legal authorization, and non-disclosure agreements.
  2. **Intelligence Gathering (Reconnaissance / OSINT)**: Passive domain profiling (whois, DNS records via `dig`/`nslookup`), employee information, search engine dorking, Shodan infrastructure queries.
  3. **Threat Modeling & Vulnerability Scanning**: Active port scanning (`nmap`), banner grabbing, web application directory busting (`gobuster`, `ffuf`), service vulnerability matching.
  4. **Vulnerability Analysis & Exploitation Theory**: Validating security flaws in controlled, authorized test environments without causing denial-of-service or business interruption.
  5. **Post-Exploitation & Privilege Escalation**: Demonstrating business impact, analyzing access boundaries, and proving risk.
  6. **Reporting & Remediation**: Comprehensive executive summary, technical vulnerability findings, CVSS v3 risk ratings, and actionable remediation steps.

- **Essential Security Assessment Tools**:
  - **Burp Suite / OWASP ZAP**: Intercepting HTTP/HTTPS proxy to inspect, manipulate, and replay web requests, audit parameters, and test authentication logic.
  - **Nmap**: Network mapping, host discovery, OS detection, and vulnerability scripting (NSE).
  - **Wireshark / TShark**: Packet sniffer for analyzing PCAP network traffic, unencrypted protocols (HTTP, FTP, Telnet), and TLS handshakes.
  - **Ghidra / IDA Free**: Reverse engineering frameworks for disassembling and decompiling compiled x86/ARM executables.
  - **Hashcat / John the Ripper**: Auditing password hash security, evaluating password complexity policies.

- **Web Application Vulnerabilities & Remediation**:
  - **Insecure Direct Object Reference (IDOR)**: Missing access control checks where changing an ID parameter (e.g., `/api/user?id=102` to `101`) displays another user's private data. Remediation: Enforce server-side session authorization on every object request.
  - **Cross-Site Scripting (XSS)**: Untrusted data rendered directly into browser DOM. Remediation: Context-aware HTML entity encoding, Content Security Policy (CSP), avoiding `innerHTML` in frontend frameworks.
  - **Server-Side Request Forgery (SSRF)**: Vulnerable server fetches a user-supplied URL, allowing attackers to access internal cloud metadata (`169.254.169.254`) or localhost services. Remediation: Strict URL whitelisting and blocking private IP ranges.

- **Privilege Escalation Mechanics**:
  - **Linux Privilege Escalation**:
    - SUID binaries with misconfigured execution permissions (`find / -perm -u=s -type f 2>/dev/null`).
    - Misconfigured sudo privileges (`sudo -l` entries permitting execution without password).
    - World-writable cron jobs or PATH hijacking in root scripts.
  - **Windows Privilege Escalation**:
    - Unquoted Service Paths containing spaces in the path without quotes.
    - Weak service permissions allowing non-admin users to modify the binary path of a service running as `NT AUTHORITY\SYSTEM`.
    - Token Impersonation (`SeImpersonatePrivilege` or `SeAssignPrimaryTokenPrivilege`) abused via named pipe potato exploits (PrintSpoofer, GodPotato).
    - `AlwaysInstallElevated` registry policy allowing regular users to run MSI installation packages with elevated privileges.

---

## 8. Threat Hunting, SIEM & Defensive Blue Teaming

- **MITRE ATT&CK Framework**:
  - Globally accessible knowledge base of adversary tactics, techniques, and procedures (TTPs) based on real-world observations.
  - Matrix Tactics: Initial Access -> Execution -> Persistence -> Privilege Escalation -> Defense Evasion -> Credential Access -> Discovery -> Lateral Movement -> Collection -> Command and Control (C2) -> Exfiltration -> Impact.
- **SIEM & Log Correlation (Splunk SPL)**:
  - Detection of Brute Force / Excessive Failed Logons:
    `index=wineventlog EventCode=4625 | stats count by user, src_ip | where count > 10`
  - Detection of Suspicious Process Creation (Command Prompt from Office Applications):
    `index=wineventlog EventCode=4688 ParentProcessName="*winword.exe" OR ParentProcessName="*excel.exe" ProcessName="*cmd.exe" OR ProcessName="*powershell.exe"`
  - Detection of LSASS Memory Access (Mimikatz):
    `index=sysmon EventCode=10 TargetImage="*lsass.exe" GrantedAccess="0x1010" OR GrantedAccess="0x143a"`
