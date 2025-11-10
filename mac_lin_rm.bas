Option Explicit

Sub Update_QA_Metrics()
    Dim wb As Workbook
    Dim wsSrc As Worksheet, wsRep As Worksheet
    Dim loQA As ListObject
    Dim colComp As Long, colPF As Long, colTick As Long, colRev As Long, colCSA As Long
    Dim i As Long, dt As Date, minDate As Date, maxDate As Date, haveDate As Boolean
    Dim periodText As String
    Dim totalCount As Long, passCount As Long, failCount As Long
    Dim dictTick As Object, dictTickPass As Object
    Dim dictRev As Object, dictRevPass As Object
    Dim dictCSA As Object, dictCSAPass As Object
    Dim pf As String, tick As String, rev As String, csa As String
    
    Set wb = ThisWorkbook
    Set wsSrc = wb.Worksheets("QA Sample Set")
    Set loQA = wsSrc.ListObjects("QA_Sam")
    
    On Error Resume Next
    Set wsRep = wb.Worksheets("Reporting_Metrics")
    On Error GoTo 0
    If wsRep Is Nothing Then
        Set wsRep = wb.Worksheets.Add
        wsRep.Name = "Reporting_Metrics"
    End If
    
    Application.ScreenUpdating = False
    
    '=== Identify required columns ===
    colComp = GetListColumnIndex(loQA, Array("Completed Date"))
    colPF = GetListColumnIndex(loQA, Array("Pass/Fail"))
    colTick = GetListColumnIndex(loQA, Array("Tickler Type"))
    colRev = GetListColumnIndex(loQA, Array("Reviewer"))
    colCSA = GetListColumnIndex(loQA, Array("Offshore CSA (Completed by)", "Offshore CSA"))
    
    If colComp = 0 Or colPF = 0 Then
        MsgBox "Missing 'Completed Date' or 'Pass/Fail' column in QA_Sam.", vbCritical
        GoTo Cleanup
    End If
    
    '=== Determine QA period ===
    For i = 1 To loQA.ListRows.Count
        If Trim(loQA.DataBodyRange.Cells(i, colPF).Value) <> "" And _
           IsDate(loQA.DataBodyRange.Cells(i, colComp).Value) Then
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
        If Trim(periodText) = "" Then GoTo Cleanup
    End If
    
    '=== Build dictionaries ===
    Set dictTick = CreateObject("Scripting.Dictionary")
    Set dictTickPass = CreateObject("Scripting.Dictionary")
    Set dictRev = CreateObject("Scripting.Dictionary")
    Set dictRevPass = CreateObject("Scripting.Dictionary")
    Set dictCSA = CreateObject("Scripting.Dictionary")
    Set dictCSAPass = CreateObject("Scripting.Dictionary")
    
    For i = 1 To loQA.ListRows.Count
        pf = LCase(Trim(loQA.DataBodyRange.Cells(i, colPF).Value))
        If pf <> "" Then
            totalCount = totalCount + 1
            If pf = "pass" Then passCount = passCount + 1
            If pf = "fail" Then failCount = failCount + 1
        End If
        
        If colTick > 0 Then
            tick = Trim(loQA.DataBodyRange.Cells(i, colTick).Value)
            If tick <> "" Then
                dictTick(tick) = dictTick(tick) + 1
                If pf = "pass" Then dictTickPass(tick) = dictTickPass(tick) + 1
            End If
        End If
        
        If colRev > 0 Then
            rev = Trim(loQA.DataBodyRange.Cells(i, colRev).Value)
            If rev <> "" Then
                dictRev(rev) = dictRev(rev) + 1
                If pf = "pass" Then dictRevPass(rev) = dictRevPass(rev) + 1
            End If
        End If
        
        If colCSA > 0 Then
            csa = Trim(loQA.DataBodyRange.Cells(i, colCSA).Value)
            If csa <> "" Then
                dictCSA(csa) = dictCSA(csa) + 1
                If pf = "pass" Then dictCSAPass(csa) = dictCSAPass(csa) + 1
            End If
        End If
    Next i
    
    '=== 1) Summary_Stats (Vertical) ===
    Dim summaryLastRow As Long
    summaryLastRow = UpdateSummaryTable_Vertical(wsRep, periodText, totalCount, passCount, failCount, 7)
    
    '=== Anchor next tables to the right ===
    Dim loSummary As ListObject
    Set loSummary = wsRep.ListObjects("Summary_Stats")
    Dim nextAnchorCol As Long
    nextAnchorCol = loSummary.Range.Column + loSummary.Range.Columns.Count + 2
    
    '=== 2) Tickler table ===
    Dim ticklerEndCol As Long
    ticklerEndCol = UpdateBreakdown_SideBySide(wsRep, "Tickler_Type_Metrics", "Tickler Type", _
                                               periodText, dictTick, dictTickPass, loSummary.HeaderRowRange.Row, nextAnchorCol)
    
    '=== 3) Reviewer table ===
    Dim reviewerAnchorCol As Long
    reviewerAnchorCol = ticklerEndCol + 2
    Dim reviewerEndCol As Long
    reviewerEndCol = UpdateBreakdown_SideBySide(wsRep, "Reviewer_Metrics", "Reviewer", _
                                                periodText, dictRev, dictRevPass, loSummary.HeaderRowRange.Row, reviewerAnchorCol)
    
    '=== 4) CSA table ===
    Dim csaAnchorCol As Long
    csaAnchorCol = reviewerEndCol + 2
    Call UpdateBreakdown_SideBySide(wsRep, "Offshore_CSA_Metrics", "Offshore CSA (Completed by)", _
                                    periodText, dictCSA, dictCSAPass, loSummary.HeaderRowRange.Row, csaAnchorCol)
    
    wsRep.Columns.AutoFit
    MsgBox "QA Metrics updated for: " & periodText, vbInformation

Cleanup:
    Application.ScreenUpdating = True
End Sub


'==================== SUMMARY (VERTICAL, ROW PER PERIOD) ====================

Private Function UpdateSummaryTable_Vertical(ws As Worksheet, periodText As String, _
                                             totalCount As Long, passCount As Long, failCount As Long, _
                                             anchorRow As Long) As Long
    Dim lo As ListObject
    Dim passPct As Double, failPct As Double
    Dim foundCell As Range
    Dim lastRow As Long
    
    passPct = IIf(totalCount > 0, passCount / totalCount, 0)
    failPct = IIf(totalCount > 0, failCount / totalCount, 0)
    
    On Error Resume Next
    Set lo = ws.ListObjects("Summary_Stats")
    On Error GoTo 0
    
    ' Create the table if missing
    If lo Is Nothing Then
        ws.Range("A" & anchorRow).Resize(1, 6).Value = _
            Array("Period", "Total QA Reviewed", "Passed", "Failed", "Pass %", "Fail %")
        Set lo = ws.ListObjects.Add(xlSrcRange:=ws.Range("A" & anchorRow & ":F" & anchorRow), _
                                    XlListObjectHasHeaders:=xlYes)
        lo.Name = "Summary_Stats"
    End If
    
    ' Find existing period row
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    Set foundCell = Nothing
    On Error Resume Next
    Set foundCell = ws.Range("A" & (anchorRow + 1) & ":A" & lastRow).Find(What:=periodText, LookIn:=xlValues, LookAt:=xlWhole)
    On Error GoTo 0
    
    ' If not found, append
    If foundCell Is Nothing Then
        lastRow = lastRow + 1
        ws.Cells(lastRow, "A").Value = periodText
        Set foundCell = ws.Cells(lastRow, "A")
    End If
    
    ' Write data
    With ws
        .Cells(foundCell.Row, "B").Value = totalCount
        .Cells(foundCell.Row, "C").Value = passCount
        .Cells(foundCell.Row, "D").Value = failCount
        .Cells(foundCell.Row, "E").Value = passPct
        .Cells(foundCell.Row, "F").Value = failPct
    End With
    
    ws.Range("B" & foundCell.Row & ":D" & foundCell.Row).NumberFormat = "0"
    ws.Range("E" & foundCell.Row & ":F" & foundCell.Row).NumberFormat = "0%"
    
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    lo.Resize ws.Range("A" & anchorRow & ":F" & lastRow)
    
    UpdateSummaryTable_Vertical = lastRow
End Function


'==================== BREAKDOWN TABLES (SIDE BY SIDE, COLS PER PERIOD) ====================

Private Function UpdateBreakdown_SideBySide(ws As Worksheet, tblName As String, firstColHeader As String, _
                                            periodText As String, dictAll As Object, dictPass As Object, _
                                            anchorRow As Long, anchorCol As Long) As Long
    Dim lo As ListObject, colCount As Long, colPct As Long
    Dim key As Variant, body As Range, rCell As Range
    Dim hdrCell As Range, lastCol As Long
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    ' Create new table if missing
    If lo Is Nothing Then
        Set hdrCell = ws.Cells(anchorRow, anchorCol)
        hdrCell.Value = firstColHeader
        hdrCell.Offset(0, 1).Value = periodText & " - Count"
        hdrCell.Offset(0, 2).Value = periodText & " - Pass %"
        Set lo = ws.ListObjects.Add(xlSrcRange:=ws.Range(hdrCell, hdrCell.Offset(0, 2)), _
                                    XlListObjectHasHeaders:=xlYes)
        lo.Name = tblName
    End If
    
    ' Ensure columns for period
    colCount = FindHeaderColumn(lo, periodText & " - Count")
    colPct = FindHeaderColumn(lo, periodText & " - Pass %")
    If colCount = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Count"
        colCount = lo.ListColumns.Count
    End If
    If colPct = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Pass %"
        colPct = lo.ListColumns.Count
    End If
    
    ' Ensure all keys as rows
    On Error Resume Next
    Set body = lo.ListColumns(1).DataBodyRange
    On Error GoTo 0
    For Each key In dictAll.Keys
        If body Is Nothing Or IsError(Application.Match(key, body, 0)) Then
            lo.ListRows.Add
            On Error Resume Next
            Set body = lo.ListColumns(1).DataBodyRange
            On Error GoTo 0
            body.Cells(body.Rows.Count, 1).Value = key
        End If
    Next key
    
    ' Fill data
    If Not body Is Nothing Then
        For Each rCell In body.Cells
            key = CStr(rCell.Value)
            If dictAll.exists(key) Then
                Dim total As Long, passed As Long, pct As Double
                total = dictAll(key)
                passed = IIf(dictPass.exists(key), dictPass(key), 0)
                pct = IIf(total > 0, passed / total, 0)
                rCell.Offset(0, colCount - 1).Value = total
                rCell.Offset(0, colPct - 1).Value = pct
            End If
        Next rCell
    End If
    
    lo.ListColumns(colCount).DataBodyRange.NumberFormat = "0"
    lo.ListColumns(colPct).DataBodyRange.NumberFormat = "0%"
    
    lastCol = lo.Range.Column + lo.Range.Columns.Count - 1
    UpdateBreakdown_SideBySide = lastCol
End Function


'==================== HELPERS ====================

Private Function GetListColumnIndex(lo As ListObject, names As Variant) As Long
    Dim lc As ListColumn, nm As Variant
    For Each nm In names
        For Each lc In lo.ListColumns
            If StrComp(Trim$(CStr(lc.Name)), Trim$(CStr(nm)), vbTextCompare) = 0 Then
                GetListColumnIndex = lc.Index
                Exit Function
            End If
        Next lc
    Next nm
End Function

Private Function FindHeaderColumn(lo As ListObject, headerName As String) As Long
    Dim lc As ListColumn
    For Each lc In lo.ListColumns
        If StrComp(Trim$(CStr(lc.Name)), Trim$(CStr(headerName)), vbTextCompare) = 0 Then
            FindHeaderColumn = lc.Index
            Exit Function
        End If
    Next lc
    FindHeaderColumn = 0
End Function