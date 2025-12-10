Private Sub Worksheet_Change(ByVal Target As Range)
    On Error GoTo CleanExit

    Dim dv As Validation
    Dim src As String
    Dim newVal As String
    Dim oldVal As String

    ' Only handle single-cell changes
    If Target.CountLarge > 1 Then Exit Sub

    ' Try to get validation object for the changed cell
    On Error Resume Next
    Set dv = Target.Validation
    On Error GoTo CleanExit

    ' If there is no validation, or it's not a list, ignore
    If dv Is Nothing Then Exit Sub
    If dv.Type <> xlValidateList Then Exit Sub

    ' Check that the validation source is Keys!$M$3:$M$21
    src = dv.Formula1          ' e.g. "=Keys!$M$3:$M$21" or "='Keys'!$M$3:$M$21"
    src = Replace(src, "=", "")
    src = Replace(src, "'", "")

    If StrComp(src, "Keys!$M$3:$M$21", vbTextCompare) <> 0 Then
        ' Different DV list – do nothing
        Exit Sub
    End If

    ' New value user just selected
    newVal = Target.Value

    ' Allow user to clear the cell normally
    If Len(newVal) = 0 Then Exit Sub

    ' From here on, we are going to rewrite the cell, so disable events
    Application.EnableEvents = False

    ' Use Undo to retrieve the old value of the cell
    Application.Undo
    oldVal = Target.Value

    ' If the cell was empty before, just put the new value
    If Len(oldVal) = 0 Then
        Target.Value = newVal
    Else
        ' Prevent duplicates: if newVal already exists, keep old list
        If InStr(1, oldVal, newVal, vbTextCompare) > 0 Then
            Target.Value = oldVal
        Else
            Target.Value = oldVal & ", " & newVal   ' change separator if you want
        End If
    End If

CleanExit:
    On Error Resume Next
    Application.EnableEvents = True
End Sub