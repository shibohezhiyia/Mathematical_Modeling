Set UAC = CreateObject("Shell.Application")
UAC.ShellExecute "cmd.exe", "/c start_server.bat", Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")), "runas", 1
