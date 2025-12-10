Private Sub Worksheet_Change(ByVal Target As Range)
    On Error GoTo ExitHandler

    Dim rngDV As Range
    Dim v As Validation
    Dim newVal As String
    Dim oldVal As String
    Dim arr As Variant
    Dim i As Long
    Dim exists As Boolean

    '===== CONFIG: ALL DROPDOWN RANGES WITH MULTI-SELECT =====
    ' Adjust these ranges to match your dropdown cells
    Set rngDV = Union( _
        Me.Range("N3:N500"), _
        Me.Range("O3:O500"), _
        Me.Range("P3:P500"), _
        Me.Range("Q3:Q500") _
    )
    '=========================================================

    'Only act if the changed cell is inside ANY of the DV ranges
    If Intersect(Target, rngDV) Is Nothing Then Exit Sub

    'Ignore multi-cell edits (copy-paste etc.)
    If Target.CountLarge > 1 Then Exit Sub

    'Make sure the cell actually has a list validation
    On Error Resume Next
    Set v = Target.Validation
    On Error GoTo ExitHandler

    If v Is Nothing Then Exit Sub
    If v.Type <> xlValidateList Then Exit Sub

    newVal = Target.Value

    'If user cleared the cell, don't do anything special
    If Len(newVal) = 0 Then Exit Sub

    Application.EnableEvents = False

    'Use Undo to get the old content of the cell
    Application.Undo
    oldVal = Target.Value

    'If previously empty, just set the new value
    If Len(oldVal) = 0 Then
        Target.Value = newVal
    Else
        'Check if the new value already exists in the list (avoid duplicates)
        arr = Split(oldVal, ",")

        exists = False
        For i = LBound(arr) To UBound(arr)
            If Trim$(arr(i)) = newVal Then
                exists = True
                Exit For
            End If
        Next i

        If exists Then
            'Keep old list as-is
            Target.Value = oldVal
        Else
            'Append with comma + space separator
            Target.Value = oldVal & ", " & newVal
        End If
    End If

    '==== SUPPRESS GREEN ERROR TRIANGLE FOR THIS CELL ====
    On Error Resume Next
    Target.Errors(xlListDataValidation).Ignore = True
    On Error GoTo ExitHandler
    '=====================================================

ExitHandler:
    On Error Resume Next
    Application.EnableEvents = True
End Sub