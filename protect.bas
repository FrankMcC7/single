If Worksheets("QA_Parameters").ProtectContents = True Then
    MsgBox "QA_Parameters sheet is protected. Please unprotect it before running this macro.", vbCritical
    Exit Sub
End If

Sub Unprotect_QA_Parameters()
    Worksheets("QA_Parameters").Unprotect Password:="YourPasswordHere"
End Sub

Sub Protect_QA_Parameters()
    With Worksheets("QA_Parameters")
        .Protect Password:="YourPasswordHere", AllowFiltering:=True
    End With
End Sub