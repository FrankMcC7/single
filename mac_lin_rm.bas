Option Explicit

Sub Update_QA_Metrics()
    Dim wb As Workbook, wsSrc As Worksheet, wsRep As Worksheet
    Dim loQA As ListObject
    Dim periodText As String
    Dim dt As Date, minDate As Date, maxDate As Date, haveDate As Boolean
    Dim i As Long
    Dim totalCount As Long, passCount As Long, failCount As Long
    Dim dictTick As Object, dictTickPass As Object
    Dim dictRev As Object, dictRevPass As Object
    Dim dictCSA As Object, dictCSAPass As Object
    Dim tick As String, rev As String, csa As String, pf As String
    Dim rng As Range, colComp As Long
    
    Set wb = ThisWorkbook
    Set wsSrc = wb.Worksheets("QA Sample Set")
    Set wsRep = wb.Worksheets("Reporting_Metrics")
    Set loQA = wsSrc.ListObjects("QA_Sam")
    
    Application.ScreenUpdating = False
    
    '=== Determine period from Completed Date ===
    On Error Resume Next
    colComp = loQA.ListColumns("Completed Date").Index
    On Error GoTo 0
    
    If colComp = 0 Then
        MsgBox "Column 'Completed Date' not found.", vbCritical
        Exit Sub
    End If
    
    For i = 1 To loQA.ListRows.Count
        If IsDate(loQA.DataBodyRange.Cells(i, colComp).Value) Then
            dt = CDate(loQA.DataBodyRange.Cells(i, colComp).Value)
            If Not haveDate Then
                minDate = dt: maxDate = dt: haveDate = True
            Else
                If dt < minDate Then minDate = dt
                If dt > maxDate Then maxDate = dt
            End If
        End If
    Next i
    
    If haveDate Then
        periodText = Format(minDate, "mmmm yyyy")
    Else
        periodText = InputBox("Enter QA period (e.g. September 2025):", "QA Period")
        If periodText = "" Then Exit Sub
    End If
    
    '=== Build dictionaries ===
    Set dictTick = CreateObject("Scripting.Dictionary")
    Set dictTickPass = CreateObject("Scripting.Dictionary")
    Set dictRev = CreateObject("Scripting.Dictionary")
    Set dictRevPass = CreateObject("Scripting.Dictionary")
    Set dictCSA = CreateObject("Scripting.Dictionary")
    Set dictCSAPass = CreateObject("Scripting.Dictionary")
    
    Dim colTick As Long, colRev As Long, colCSA As Long, colPF As Long
    colTick = loQA.ListColumns("Tickler Type").Index
    colRev = loQA.ListColumns("Reviewer").Index
    colCSA = loQA.ListColumns("Offshore CSA (Completed by)").Index
    colPF = loQA.ListColumns("Pass/Fail").Index
    
    totalCount = 0: passCount = 0: failCount = 0
    
    For i = 1 To loQA.ListRows.Count
        pf = Trim(loQA.DataBodyRange.Cells(i, colPF).Value)
        If pf <> "" Then
            totalCount = totalCount + 1
            If LCase(pf) = "pass" Then passCount = passCount + 1
            If LCase(pf) = "fail" Then failCount = failCount + 1
        End If
        
        tick = Trim(loQA.DataBodyRange.Cells(i, colTick).Value)
        rev = Trim(loQA.DataBodyRange.Cells(i, colRev).Value)
        csa = Trim(loQA.DataBodyRange.Cells(i, colCSA).Value)
        
        ' Tickler Type
        If tick <> "" Then
            dictTick(tick) = dictTick(tick) + 1
            If LCase(pf) = "pass" Then dictTickPass(tick) = dictTickPass(tick) + 1
        End If
        ' Reviewer
        If rev <> "" Then
            dictRev(rev) = dictRev(rev) + 1
            If LCase(pf) = "pass" Then dictRevPass(rev) = dictRevPass(rev) + 1
        End If
        ' CSA
        If csa <> "" Then
            dictCSA(csa) = dictCSA(csa) + 1
            If LCase(pf) = "pass" Then dictCSAPass(csa) = dictCSAPass(csa) + 1
        End If
    Next i
    
    '=== Clear old Summary Chart area ===
    wsRep.Activate
    wsRep.Range("A1:Z1000").Font.Name = "Calibri"
    
    '--- Summary Stats at A7 ---
    UpdateSummaryTable wsRep, periodText, totalCount, passCount, failCount, 7
    
    '--- Tickler Type table next ---
    Dim nextRow As Long
    nextRow = wsRep.ListObjects("Summary_Stats").Range.Row + wsRep.ListObjects("Summary_Stats").Range.Rows.Count + 3
    UpdateBreakdown wsRep, "Tickler_Type_Table", "Tickler Type", periodText, dictTick, dictTickPass, nextRow
    
    '--- Reviewer table next ---
    nextRow = wsRep.ListObjects("Tickler_Type_Table").Range.Row + wsRep.ListObjects("Tickler_Type_Table").Range.Rows.Count + 3
    UpdateBreakdown wsRep, "Reviewer_Table", "Reviewer", periodText, dictRev, dictRevPass, nextRow
    
    '--- CSA table next ---
    nextRow = wsRep.ListObjects("Reviewer_Table").Range.Row + wsRep.ListObjects("Reviewer_Table").Range.Rows.Count + 3
    UpdateBreakdown wsRep, "CSA_Table", "Offshore CSA (Completed by)", periodText, dictCSA, dictCSAPass, nextRow
    
    Application.ScreenUpdating = True
    MsgBox "QA Metrics updated for period: " & periodText, vbInformation
End Sub


'==================== SUMMARY TABLE ====================
Private Sub UpdateSummaryTable(ws As Worksheet, periodText As String, totalCount As Long, passCount As Long, failCount As Long, anchorRow As Long)
    Dim lo As ListObject
    Dim colPeriod As Long
    Dim passPct As Double
    Dim nextCol As Long
    
    passPct = IIf(totalCount > 0, passCount / totalCount, 0)
    
    On Error Resume Next
    Set lo = ws.ListObjects("Summary_Stats")
    On Error GoTo 0
    
    ' Create if missing
    If lo Is Nothing Then
        ws.Range("A" & anchorRow).Value = "Metric"
        ws.Range("B" & anchorRow).Value = periodText & " - Count"
        ws.Range("C" & anchorRow).Value = periodText & " - Pass %"
        ws.ListObjects.Add xlSrcRange, ws.Range("A" & anchorRow & ":C" & anchorRow), , xlYes
        ws.ListObjects(ws.ListObjects.Count).Name = "Summary_Stats"
        Set lo = ws.ListObjects("Summary_Stats")
        
        ws.Range("A" & anchorRow + 1).Resize(3).Value = Application.Transpose(Array("Total QA Reviewed", "Passed", "Failed"))
    End If
    
    ' Overwrite or add new period columns
    colPeriod = FindHeaderColumn(lo, periodText & " - Count")
    If colPeriod = 0 Then
        nextCol = lo.ListColumns.Count + 1
        lo.HeaderRowRange.Cells(1, nextCol).Value = periodText & " - Count"
        lo.HeaderRowRange.Cells(1, nextCol + 1).Value = periodText & " - Pass %"
    End If
    
    ' Update values
    lo.ListColumns(periodText & " - Count").DataBodyRange.Cells(1, 1).Value = totalCount
    lo.ListColumns(periodText & " - Count").DataBodyRange.Cells(2, 1).Value = passCount
    lo.ListColumns(periodText & " - Count").DataBodyRange.Cells(3, 1).Value = failCount
    
    lo.ListColumns(periodText & " - Pass %").DataBodyRange.Cells(1, 1).Value = ""
    lo.ListColumns(periodText & " - Pass %").DataBodyRange.Cells(2, 1).Value = passPct
    lo.ListColumns(periodText & " - Pass %").DataBodyRange.Cells(3, 1).Value = 1 - passPct
    
    lo.ListColumns(periodText & " - Count").DataBodyRange.NumberFormat = "0"
    lo.ListColumns(periodText & " - Pass %").DataBodyRange.NumberFormat = "0%"
End Sub


'==================== BREAKDOWN TABLES ====================
Private Sub UpdateBreakdown(ws As Worksheet, tblName As String, firstColHeader As String, _
                            periodText As String, dictAll As Object, dictPass As Object, anchorRow As Long)
    Dim lo As ListObject
    Dim key As Variant
    Dim colCount As Long, colPct As Long
    Dim rowAdd As Long
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    ' Create table if not present
    If lo Is Nothing Then
        ws.Range("A" & anchorRow).Value = firstColHeader
        ws.Range("B" & anchorRow).Value = periodText & " - Count"
        ws.Range("C" & anchorRow).Value = periodText & " - Pass %"
        ws.ListObjects.Add xlSrcRange, ws.Range("A" & anchorRow & ":C" & anchorRow), , xlYes
        ws.ListObjects(ws.ListObjects.Count).Name = tblName
        Set lo = ws.ListObjects(tblName)
    End If
    
    ' Overwrite existing period columns or add new
    colCount = FindHeaderColumn(lo, periodText & " - Count")
    colPct = FindHeaderColumn(lo, periodText & " - Pass %")
    If colCount = 0 Then
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count + 1).Value = periodText & " - Count"
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count + 2).Value = periodText & " - Pass %"
        colCount = FindHeaderColumn(lo, periodText & " - Count")
        colPct = FindHeaderColumn(lo, periodText & " - Pass %")
    End If
    
    ' Ensure all keys exist as rows
    For Each key In dictAll.Keys
        Dim found As Range
        Set found = lo.ListColumns(1).DataBodyRange.Find(What:=key, LookIn:=xlValues, LookAt:=xlWhole)
        If found Is Nothing Then
            lo.ListRows.Add
            lo.ListColumns(1).DataBodyRange.Cells(lo.ListRows.Count).Value = key
        End If
    Next key
    
    ' Populate data
    For Each key In dictAll.Keys
        Dim r As Range
        Set r = lo.ListColumns(1).DataBodyRange.Find(What:=key, LookIn:=xlValues, LookAt:=xlWhole)
        If Not r Is Nothing Then
            Dim total As Long, passed As Long
            total = dictAll(key)
            passed = 0
            If dictPass.exists(key) Then passed = dictPass(key)
            
            lo.DataBodyRange.Cells(r.Row - lo.HeaderRowRange.Row, colCount).Value = total
            lo.DataBodyRange.Cells(r.Row - lo.HeaderRowRange.Row, colPct).Value = IIf(total > 0, passed / total, 0)
        End If
    Next key
    
    lo.ListColumns(periodText & " - Count").DataBodyRange.NumberFormat = "0"
    lo.ListColumns(periodText & " - Pass %").DataBodyRange.NumberFormat = "0%"
End Sub


'==================== HELPER FUNCTIONS ====================
Private Function FindHeaderColumn(lo As ListObject, headerName As String) As Long
    Dim lc As ListColumn
    For Each lc In lo.ListColumns
        If StrComp(Trim(lc.Name), Trim(headerName), vbTextCompare) = 0 Then
            FindHeaderColumn = lc.Index
            Exit Function
        End If
    Next lc
    FindHeaderColumn = 0
End Function