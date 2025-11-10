Option Explicit

Sub Update_QA_Metrics()
    Dim wb As Workbook
    Dim wsSrc As Worksheet, wsRep As Worksheet
    Dim loQA As ListObject
    Dim colComp As Long, colPF As Long, colTick As Long, colRev As Long, colCSA As Long
    Dim i As Long
    Dim dt As Date, minDate As Date, maxDate As Date, haveDate As Boolean
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
    
    '--- Identify columns ---
    colComp = GetListColumnIndex(loQA, Array("Completed Date"))
    colPF = GetListColumnIndex(loQA, Array("Pass/Fail"))
    colTick = GetListColumnIndex(loQA, Array("Tickler Type"))
    colRev = GetListColumnIndex(loQA, Array("Reviewer"))
    colCSA = GetListColumnIndex(loQA, Array("Offshore CSA (Completed by)", "Offshore CSA"))
    
    If colComp = 0 Or colPF = 0 Then
        MsgBox "Missing 'Completed Date' or 'Pass/Fail' column in QA_Sam.", vbCritical
        GoTo Cleanup
    End If
    
    '--- Determine period from Completed Date where Pass/Fail present ---
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
        If Month(minDate) = Month(maxDate) And Year(minDate) = Year(maxDate) Then
            periodText = Format(minDate, "mmmm yyyy")
        Else
            periodText = Format(minDate, "mmmm yyyy") & " - " & Format(maxDate, "mmmm yyyy")
        End If
    Else
        periodText = InputBox("Couldn't detect period from 'Completed Date'." & vbCrLf & _
                              "Enter period (e.g. September 2025):", "QA Period")
        If Trim(periodText) = "" Then GoTo Cleanup
    End If
    
    '--- Init dictionaries ---
    Set dictTick = CreateObject("Scripting.Dictionary")
    Set dictTickPass = CreateObject("Scripting.Dictionary")
    Set dictRev = CreateObject("Scripting.Dictionary")
    Set dictRevPass = CreateObject("Scripting.Dictionary")
    Set dictCSA = CreateObject("Scripting.Dictionary")
    Set dictCSAPass = CreateObject("Scripting.Dictionary")
    
    '--- Aggregate data ---
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
    
    '--- 1) Update Summary_Stats (vertical) at A7 ---
    Dim summaryLastRow As Long
    summaryLastRow = UpdateSummaryTable_Vertical(wsRep, periodText, totalCount, passCount, failCount, 7)
    
    ' Get Summary table for positioning others
    Dim loSummary As ListObject
    Set loSummary = wsRep.ListObjects("Summary_Stats")
    Dim nextAnchorCol As Long
    nextAnchorCol = loSummary.Range.Column + loSummary.Range.Columns.Count + 2  ' 1 table gap
    
    '--- 2) Tickler table to the right of Summary ---
    Dim ticklerEndCol As Long
    ticklerEndCol = UpdateBreakdown_SideBySide(wsRep, "Tickler_Type_Metrics", "Tickler Type", _
                                               periodText, dictTick, dictTickPass, loSummary.HeaderRowRange.Row, nextAnchorCol)
    
    '--- 3) Reviewer table to the right of Tickler ---
    Dim reviewerAnchorCol As Long
    reviewerAnchorCol = ticklerEndCol + 2
    Dim reviewerEndCol As Long
    reviewerEndCol = UpdateBreakdown_SideBySide(wsRep, "Reviewer_Metrics", "Reviewer", _
                                                periodText, dictRev, dictRevPass, loSummary.HeaderRowRange.Row, reviewerAnchorCol)
    
    '--- 4) CSA table to the right of Reviewer ---
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
    
    ' Create table if missing
    If lo Is Nothing Then
        ws.Range("A" & anchorRow).Resize(1, 6).Value = _
            Array("Period", "Total QA Reviewed", "Passed", "Failed", "Pass %", "Fail %")
        Set lo = ws.ListObjects.Add(xlSrcRange:=ws.Range("A" & anchorRow & ":F" & anchorRow), _
                                    XlListObjectHasHeaders:=xlYes)
        lo.Name = "Summary_Stats"
    End If
    
    ' Find existing period row
    lastRow = ws.Cells(ws.Rows.Count, lo.Range.Columns(1).Column).End(xlUp).Row
    Set foundCell = Nothing
    On Error Resume Next
    Set foundCell = ws.Range("A" & (lo.HeaderRowRange.Row + 1) & ":A" & lastRow) _
                        .Find(What:=periodText, LookIn:=xlValues, LookAt:=xlWhole)
    On Error GoTo 0
    
    ' If not found, append at bottom
    If foundCell Is Nothing Then
        lastRow = lastRow + 1
        ws.Cells(lastRow, "A").Value = periodText
        Set foundCell = ws.Cells(lastRow, "A")
    End If
    
    ' Write data into that row
    With ws
        .Cells(foundCell.Row, "B").Value = totalCount
        .Cells(foundCell.Row, "C").Value = passCount
        .Cells(foundCell.Row, "D").Value = failCount
        .Cells(foundCell.Row, "E").Value = passPct
        .Cells(foundCell.Row, "F").Value = failPct
    End With
    
    ' Format
    ws.Range("B" & foundCell.Row & ":D" & foundCell.Row).NumberFormat = "0"
    ws.Range("E" & foundCell.Row & ":F" & foundCell.Row).NumberFormat = "0%"
    
    ' Resize table to include new rows
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    lo.Resize ws.Range("A" & anchorRow & ":F" & lastRow)
    
    UpdateSummaryTable_Vertical = lastRow
End Function

'==================== BREAKDOWN TABLES (SIDE BY SIDE, COLS PER PERIOD) ====================

' Creates/updates breakdown table.
' - firstColHeader: dimension name (Tickler Type / Reviewer / Offshore CSA...)
' - anchorRow, anchorCol: where to place the header if creating new
' - Returns last used column of this table (for placing the next table)
Private Function UpdateBreakdown_SideBySide(ws As Worksheet, tblName As String, firstColHeader As String, _
                                            periodText As String, dictAll As Object, dictPass As Object, _
                                            anchorRow As Long, anchorCol As Long) As Long
    Dim lo As ListObject
    Dim colCount As Long, colPct As Long
    Dim key As Variant
    Dim body As Range
    Dim rCell As Range
    Dim hdrRange As Range
    Dim firstCol As Long, lastCol As Long
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    ' Create new table if it doesn't exist
    If lo Is Nothing Then
        Set hdrRange = ws.Cells(anchorRow, anchorCol)
        hdrRange.Value = firstColHeader
        hdrRange.Offset(0, 1).Value = periodText & " - Count"
        hdrRange.Offset(0, 2).Value = periodText & " - Pass %"
        
        Set lo = ws.ListObjects.Add(xlSrcRange:=ws.Range(hdrRange, hdrRange.Offset(0, 2)), _
                                    XlListObjectHasHeaders:=xlYes)
        lo.Name = tblName
    End If
    
    firstCol = lo.Range.Column
    lastCol = lo.Range.Column + lo.Range.Columns.Count - 1
    
    ' Ensure columns for this period exist
    colCount = FindHeaderColumn(lo, periodText & " - Count")
    colPct = FindHeaderColumn(lo, periodText & " - Pass %")
    
    If colCount = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Count"
        colCount = lo.ListColumns.Count
        lastCol = lo.Range.Column + lo.Range.Columns.Count - 1
    End If
    
    If colPct = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Pass %"
        colPct = lo.ListColumns.Count
        lastCol = lo.Range.Column + lo.Range.Columns.Count - 1
    End If
    
    ' Refresh body
    On Error Resume Next
    Set body = lo.ListColumns(1).DataBodyRange
    On Error GoTo 0
    
    ' Ensure all keys exist as rows
    For Each key In dictAll.Keys
        If Not body Is Nothing Then
            Set rCell = body.Find(What:=key, LookIn:=xlValues, LookAt:=xlWhole)
        Else
            Set rCell = Nothing
        End If
        
        If rCell Is Nothing Then
            lo.ListRows.Add
            On Error Resume Next
            Set body = lo.ListColumns(1).DataBodyRange
            On Error GoTo 0
            body.Cells(body.Rows.Count, 1).Value = key
        End If
    Next key
    
    ' Refresh body again (in case rows were added)
    On Error Resume Next
    Set body = lo.ListColumns(1).DataBodyRange
    On Error GoTo 0
    If body Is Nothing Then
        UpdateBreakdown_SideBySide = lastCol
        Exit Function
    End If
    
    ' Fill values for this period
    For Each rCell In body.Cells
        key = CStr(rCell.Value)
        If dictAll.exists(key) Then
            Dim total As Long, passed As Long, pct As Double
            total = dictAll(key)
            If dictPass.exists(key) Then
                passed = dictPass(key)
            Else
                passed = 0
            End If
            If total > 0 Then pct = passed / total Else pct = 0
            
            rCell.Offset(0, colCount - 1).Value = total
            rCell.Offset(0, colPct - 1).Value = pct
        End If
    Next rCell
    
    ' Formatting
    lo.ListColumns(colCount).DataBodyRange.NumberFormat = "0"
    lo.ListColumns(colPct).DataBodyRange.NumberFormat = "0%"
    
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