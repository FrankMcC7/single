If Worksheets("QA_Parameters").ProtectContents = True Then
    MsgBox "QA_Parameters sheet is protected. Please unprotect it before running this macro.", vbCritical
    Exit Sub
End If