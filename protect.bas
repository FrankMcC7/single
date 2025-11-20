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


Option Explicit

'====================================================================================
' SHEET PROTECTION BUTTONS FOR "QA_Parameters"  (FIXED PASSWORD VERSION)
'
' HOW THIS WORKS
'   • Change QA_PARAMS_PASSWORD below to whatever password you want.
'   • User must type this exact password when clicking:
'       - Protect button  → protects sheet if password matches the fixed one.
'       - Unprotect button → unprotects sheet if password matches the fixed one.
'
' WHERE TO SET / CHANGE PASSWORD
'   • Edit the line:
'         Const QA_PARAMS_PASSWORD As String = "YourFixedPasswordHere"
'====================================================================================

' <<< SET YOUR FIXED PASSWORD HERE >>>
Private Const QA_PARAMS_PASSWORD As String = "QAparams@123"   ' <-- change this string


'------------------------------------------------------------------------------------
' BUTTON 1: Protect "QA_Parameters" (asks user for password, compares to fixed one)
'------------------------------------------------------------------------------------
Public Sub Protect_QA_Parameters_Button()
    Dim ws As Worksheet
    Dim userPwd As String

    On Error GoTo ErrHandler
    Set ws = ThisWorkbook.Worksheets("QA_Parameters")

    ' Already protected?
    If ws.ProtectContents Then
        MsgBox "'QA_Parameters' sheet is already protected.", vbInformation
        Exit Sub
    End If

    ' Ask user for password (must match fixed one)
    userPwd = InputBox("Enter password to PROTECT 'QA_Parameters' sheet:", _
                       "Protect QA_Parameters")

    If userPwd = "" Then
        MsgBox "No password entered. Protection cancelled.", vbExclamation
        Exit Sub
    End If

    ' Compare typed password with fixed password
    If userPwd <> QA_PARAMS_PASSWORD Then
        MsgBox "Incorrect password. Protection cancelled.", vbCritical
        Exit Sub
    End If

    ' Apply protection using the fixed password constant
    ws.Protect Password:=QA_PARAMS_PASSWORD, _
               DrawingObjects:=True, _
               Contents:=True, _
               Scenarios:=True, _
               AllowFiltering:=True

    MsgBox "'QA_Parameters' sheet has been protected.", vbInformation
    Exit Sub

ErrHandler:
    MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical, "Protect_QA_Parameters_Button"
End Sub


'------------------------------------------------------------------------------------
' BUTTON 2: Unprotect "QA_Parameters" (asks user for password, compares to fixed one)
'------------------------------------------------------------------------------------
Public Sub Unprotect_QA_Parameters_Button()
    Dim ws As Worksheet
    Dim userPwd As String

    On Error GoTo ErrHandler
    Set ws = ThisWorkbook.Worksheets("QA_Parameters")

    ' Already unprotected?
    If Not ws.ProtectContents Then
        MsgBox "'QA_Parameters' sheet is already unprotected.", vbInformation
        Exit Sub
    End If

    ' Ask user for password (must match fixed one)
    userPwd = InputBox("Enter password to UNPROTECT 'QA_Parameters' sheet:", _
                       "Unprotect QA_Parameters")

    If userPwd = "" Then
        MsgBox "No password entered. Unprotect cancelled.", vbExclamation
        Exit Sub
    End If

    ' Compare typed password with fixed password
    If userPwd <> QA_PARAMS_PASSWORD Then
        MsgBox "Incorrect password. Sheet is still protected.", vbCritical
        Exit Sub
    End If

    ' Try to unprotect using the fixed password constant
    On Error Resume Next
    ws.Unprotect Password:=QA_PARAMS_PASSWORD
    On Error GoTo ErrHandler

    If ws.ProtectContents Then
        MsgBox "Unprotect failed. Sheet is still protected.", vbCritical
    Else
        MsgBox "'QA_Parameters' sheet has been unprotected.", vbInformation
    End If

    Exit Sub

ErrHandler:
    MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical, "Unprotect_QA_Parameters_Button"
End Sub