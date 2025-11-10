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
    
    ' Ensure Reporting_Metrics sheet
    On Error Resume Next
    Set wsRep = wb.Worksheets("Reporting_Metrics")
    On Error GoTo 0
    If wsRep Is Nothing Then
        Set wsRep = wb.Worksheets.Add
        wsRep.Name = "Reporting_Metrics"
    End If
    
    Application.ScreenUpdating = False
    
    '=== Identify columns ===
    colComp = GetListColumnIndex(loQA, Array("Completed Date"))
    colPF = GetListColumnIndex(loQA, Array("Pass/Fail"))
    colTick = GetListColumnIndex(loQA, Array("Tickler Type"))
    colRev = GetListColumnIndex(loQA, Array("Reviewer"))
    colCSA = GetListColumnIndex(loQA, Array("Offshore CSA (Completed by)", "Offshore CSA"))
    
    If colComp = 0 Or colPF = 0 Then
        MsgBox "Missing 'Completed Date' or 'Pass/Fail' column in QA_Sam.", vbCritical
        GoTo Cleanup
    End If
    
    '=== Determine period from Completed Date where Pass/Fail present ===
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
        ' Use month/year of minDate (your original intent)
        periodText = Format(minDate, "mmmm yyyy")
    Else
        periodText = InputBox("Couldn't detect period from 'Completed Date'." & vbCrLf & _
                              "Enter period (e.g. September 2025):", "QA Period")
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
    
    '=== 1) Summary_Stats vertical at A7 ===
    Dim summaryLastRow As Long
    summaryLastRow = UpdateSummaryTable_Vertical(wsRep, periodText, totalCount, passCount, failCount, 7)
    
    ' Get Summary_Stats table for anchor
    Dim loSummary As ListObject
    Set loSummary = wsRep.ListObjects("Summary_Stats")
    
    '=== 2) Tickler table to the right of Summary ===
    Dim ticklerAnchorCol As Long, ticklerEndCol As Long
    ticklerAnchorCol = loSummary.Range.Column + loSummary.Range.Columns.Count + 2
    ticklerEndCol = UpdateBreakdown_SideBySide( _
                        wsRep, "Tickler_Type_Metrics", "Tickler Type", _
                        periodText, dictTick, dictTickPass, _
                        loSummary.HeaderRowRange.Row, ticklerAnchorCol)
    
    '=== 3) Reviewer table to the right of Tickler ===
    Dim reviewerAnchorCol As Long, reviewerEndCol As Long
    reviewerAnchorCol = ticklerEndCol + 2
    reviewerEndCol = UpdateBreakdown_SideBySide( _
                        wsRep, "Reviewer_Metrics", "Reviewer", _
                        periodText, dictRev, dictRevPass, _
                        loSummary.HeaderRowRange.Row, reviewerAnchorCol)
    
    '=== 4) CSA table to the right of Reviewer ===
    Dim csaAnchorCol As Long
    csaAnchorCol = reviewerEndCol + 2
    Call UpdateBreakdown_SideBySide( _
            wsRep, "Offshore_CSA_Metrics", "Offshore CSA (Completed by)", _
            periodText, dictCSA, dictCSAPass, _
            loSummary.HeaderRowRange.Row, csaAnchorCol)
    
    wsRep.Columns.AutoFit
    MsgBox "QA Metrics updated for: " & periodText, vbInformation

Cleanup:
    Application.ScreenUpdating = True
End Sub


'==================== SUMMARY (VERTICAL: ONE ROW PER PERIOD) ====================

Private Function UpdateSummaryTable_Vertical(ws As Worksheet, periodText As String, _
                                             totalCount As Long, passCount As Long, failCount As Long, _
                                             anchorRow As Long) As Long
    Dim lo As ListObject
    Dim passPct As Double, failPct As Double
    Dim foundCell As Range
    Dim lastRow As Long
    Dim hdrRng As Range, tblRng As Range
    
    passPct = IIf(totalCount > 0, passCount / totalCount, 0)
    failPct = IIf(totalCount > 0, failCount / totalCount, 0)
    
    On Error Resume Next
    Set lo = ws.ListObjects("Summary_Stats")
    On Error GoTo 0
    
    ' Create if missing
    If lo Is Nothing Then
        Set hdrRng = ws.Range("A" & anchorRow)
        hdrRng.Resize(1, 6).Value = Array("Period", "Total QA Reviewed", "Passed", "Failed", "Pass %", "Fail %")
        Set tblRng = ws.Range("A" & anchorRow & ":F" & anchorRow)
        Set lo = ws.ListObjects.Add(xlSrcRange, tblRng, , xlYes)
        lo.Name = "Summary_Stats"
    End If
    
    ' Find existing period row
    lastRow = ws.Cells(ws.Rows.Count, lo.Range.Columns(1).Column).End(xlUp).Row
    If lastRow < lo.HeaderRowRange.Row + 1 Then
        ' No data rows yet
        Set foundCell = Nothing
    Else
        Set foundCell = ws.Range( _
            ws.Cells(lo.HeaderRowRange.Row + 1, lo.Range.Columns(1).Column), _
            ws.Cells(lastRow, lo.Range.Columns(1).Column) _
        ).Find(What:=periodText, LookIn:=xlValues, LookAt:=xlWhole)
    End If
    
    ' If not found, append new row
    If foundCell Is Nothing Then
        lastRow = lastRow + 1
        ws.Cells(lastRow, lo.Range.Columns(1).Column).Value = periodText
        Set foundCell = ws.Cells(lastRow, lo.Range.Columns(1).Column)
    End If
    
    ' Write values
    With ws
        .Cells(foundCell.Row, lo.Range.Columns(1).Column + 1).Value = totalCount
        .Cells(foundCell.Row, lo.Range.Columns(1).Column + 2).Value = passCount
        .Cells(foundCell.Row, lo.Range.Columns(1).Column + 3).Value = failCount
        .Cells(foundCell.Row, lo.Range.Columns(1).Column + 4).Value = passPct
        .Cells(foundCell.Row, lo.Range.Columns(1).Column + 5).Value = failPct
    End With
    
    ' Format
    ws.Range(ws.Cells(foundCell.Row, lo.Range.Columns(1).Column + 1), _
             ws.Cells(foundCell.Row, lo.Range.Columns(1).Column + 3)).NumberFormat = "0"
    ws.Range(ws.Cells(foundCell.Row, lo.Range.Columns(1).Column + 4), _
             ws.Cells(foundCell.Row, lo.Range.Columns(1).Column + 5)).NumberFormat = "0%"
    
    ' Resize table to include all used rows in this block
    lastRow = ws.Cells(ws.Rows.Count, lo.Range.Columns(1).Column).End(xlUp).Row
    Set tblRng = ws.Range( _
        ws.Cells(lo.HeaderRowRange.Row, lo.Range.Columns(1).Column), _
        ws.Cells(lastRow, lo.Range.Columns(1).Column + 5))
    lo.Resize tblRng
    
    UpdateSummaryTable_Vertical = lastRow
End Function


'==================== BREAKDOWN TABLES (SIDE BY SIDE, COLS PER PERIOD) ====================

Private Function UpdateBreakdown_SideBySide(ws As Worksheet, tblName As String, firstColHeader As String, _
                                            periodText As String, dictAll As Object, dictPass As Object, _
                                            anchorRow As Long, anchorCol As Long) As Long
    Dim lo As ListObject
    Dim hdrCell As Range, tblRng As Range
    Dim colCount As Long, colPct As Long
    Dim key As Variant
    Dim body As Range
    Dim rowIndex As Variant
    Dim lastCol As Long
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    ' Create if missing
    If lo Is Nothing Then
        Set hdrCell = ws.Cells(anchorRow, anchorCol)
        hdrCell.Value = firstColHeader
        hdrCell.Offset(0, 1).Value = periodText & " - Count"
        hdrCell.Offset(0, 2).Value = periodText & " - Pass %"
        Set tblRng = ws.Range(hdrCell, hdrCell.Offset(0, 2))
        Set lo = ws.ListObjects.Add(xlSrcRange, tblRng, , xlYes)
        lo.Name = tblName
    End If
    
    ' Ensure period columns
    colCount = FindHeaderColumn(lo, periodText & " - Count")
    If colCount = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Count"
        colCount = lo.ListColumns.Count
    End If
    
    colPct = FindHeaderColumn(lo, periodText & " - Pass %")
    If colPct = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Pass %"
        colPct = lo.ListColumns.Count
    End If
    
    ' Ensure rows for all keys
    If lo.ListRows.Count > 0 Then
        Set body = lo.ListColumns(1).DataBodyRange
    Else
        Set body = Nothing
    End If
    
    For Each key In dictAll.Keys
        If body Is Nothing Then
            lo.ListRows.Add
            Set body = lo.ListColumns(1).DataBodyRange
            body.Cells(body.Rows.Count, 1).Value = key
        Else
            rowIndex = Application.Match(CStr(key), body, 0)
            If IsError(rowIndex) Then
                lo.ListRows.Add
                Set body = lo.ListColumns(1).DataBodyRange
                body.Cells(body.Rows.Count, 1).Value = key
            End If
        End If
    Next key
    
    ' Refresh body after potential adds
    If lo.ListRows.Count > 0 Then
        Set body = lo.ListColumns(1).DataBodyRange
    Else
        Set body = Nothing
    End If
    
    ' Fill data for this period
    If Not body Is Nothing Then
        For Each key In dictAll.Keys
            rowIndex = Application.Match(CStr(key), body, 0)
            If Not IsError(rowIndex) Then
                Dim total As Long, passed As Long, pct As Double
                total = dictAll(key)
                If dictPass.exists(key) Then
                    passed = dictPass(key)
                Else
                    passed = 0
                End If
                If total > 0 Then pct = passed / total Else pct = 0
                
                body.Cells(rowIndex, colCount - 1).Value = total
                body.Cells(rowIndex, colPct - 1).Value = pct
            End If
        Next key
    End If
    
    ' Format
    If lo.ListRows.Count > 0 Then
        lo.ListColumns(colCount).DataBodyRange.NumberFormat = "0"
        lo.ListColumns(colPct).DataBodyRange.NumberFormat = "0%"
    End If
    
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