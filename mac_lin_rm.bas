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
    
    '--- Loop QA rows to aggregate ---
    For i = 1 To loQA.ListRows.Count
        pf = LCase(Trim(loQA.DataBodyRange.Cells(i, colPF).Value))
        If pf <> "" Then
            totalCount = totalCount + 1
            If pf = "pass" Then passCount = passCount + 1
            If pf = "fail" Then failCount = failCount + 1
        End If
        
        ' Tickler
        If colTick > 0 Then
            tick = Trim(loQA.DataBodyRange.Cells(i, colTick).Value)
            If tick <> "" Then
                dictTick(tick) = dictTick(tick) + 1
                If pf = "pass" Then dictTickPass(tick) = dictTickPass(tick) + 1
            End If
        End If
        
        ' Reviewer
        If colRev > 0 Then
            rev = Trim(loQA.DataBodyRange.Cells(i, colRev).Value)
            If rev <> "" Then
                dictRev(rev) = dictRev(rev) + 1
                If pf = "pass" Then dictRevPass(rev) = dictRevPass(rev) + 1
            End If
        End If
        
        ' CSA
        If colCSA > 0 Then
            csa = Trim(loQA.DataBodyRange.Cells(i, colCSA).Value)
            If csa <> "" Then
                dictCSA(csa) = dictCSA(csa) + 1
                If pf = "pass" Then dictCSAPass(csa) = dictCSAPass(csa) + 1
            End If
        End If
    Next i
    
    '--- Summary at A7 ---
    Dim summaryEndRow As Long
    summaryEndRow = UpdateSummaryTable(wsRep, periodText, totalCount, passCount, failCount, 7)
    
    '--- Tickler table below summary ---
    Dim ticklerStart As Long
    ticklerStart = summaryEndRow + 3
    Dim ticklerEndRow As Long
    ticklerEndRow = UpdateBreakdown(wsRep, "Tickler_Type_Metrics", "Tickler Type", periodText, dictTick, dictTickPass, ticklerStart)
    
    '--- Reviewer table below tickler ---
    Dim reviewerStart As Long
    reviewerStart = ticklerEndRow + 3
    Dim reviewerEndRow As Long
    reviewerEndRow = UpdateBreakdown(wsRep, "Reviewer_Metrics", "Reviewer", periodText, dictRev, dictRevPass, reviewerStart)
    
    '--- CSA table below reviewer ---
    Dim csaStart As Long
    csaStart = reviewerEndRow + 3
    Call UpdateBreakdown(wsRep, "Offshore_CSA_Metrics", "Offshore CSA (Completed by)", periodText, dictCSA, dictCSAPass, csaStart)
    
    wsRep.Columns.AutoFit
    MsgBox "QA Metrics updated for: " & periodText, vbInformation

Cleanup:
    Application.ScreenUpdating = True
End Sub


'==================== SUMMARY TABLE ====================

' Returns last used row of Summary_Stats table
Private Function UpdateSummaryTable(ws As Worksheet, periodText As String, _
                                    totalCount As Long, passCount As Long, failCount As Long, _
                                    anchorRow As Long) As Long
    Dim lo As ListObject
    Dim passPct As Double, failPct As Double
    Dim colPeriod As Long
    
    passPct = IIf(totalCount > 0, passCount / totalCount, 0)
    failPct = IIf(totalCount > 0, failCount / totalCount, 0)
    
    On Error Resume Next
    Set lo = ws.ListObjects("Summary_Stats")
    On Error GoTo 0
    
    If lo Is Nothing Then
        ' Create new Summary_Stats starting at A7
        ws.Range("A" & anchorRow).Value = "Metric"
        ws.Range("A" & anchorRow + 1).Resize(5, 1).Value = Application.Transpose( _
            Array("Total QA Reviewed", "Total Pass", "Total Fail", "Pass %", "Fail %"))
        
        ws.Range("B" & anchorRow).Value = periodText
        Dim tblRng As Range
        Set tblRng = ws.Range("A" & anchorRow & ":B" & anchorRow + 5)
        Set lo = ws.ListObjects.Add(xlSrcRange, tblRng, , xlYes)
        lo.Name = "Summary_Stats"
    End If
    
    ' Find or create column for this period
    colPeriod = FindHeaderColumn(lo, periodText)
    If colPeriod = 0 Then
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText
        colPeriod = lo.ListColumns.Count
    End If
    
    ' Write values (overwrites existing)
    With lo.DataBodyRange
        .Cells(1, colPeriod).Value = totalCount
        .Cells(2, colPeriod).Value = passCount
        .Cells(3, colPeriod).Value = failCount
        .Cells(4, colPeriod).Value = passPct
        .Cells(5, colPeriod).Value = failPct
    End With
    
    ' Formatting
    lo.DataBodyRange.Rows(1).Columns(colPeriod).NumberFormat = "0"
    lo.DataBodyRange.Rows(2).Columns(colPeriod).NumberFormat = "0"
    lo.DataBodyRange.Rows(3).Columns(colPeriod).NumberFormat = "0"
    lo.DataBodyRange.Rows(4).Columns(colPeriod).NumberFormat = "0%"
    lo.DataBodyRange.Rows(5).Columns(colPeriod).NumberFormat = "0%"
    
    UpdateSummaryTable = lo.Range.Rows(lo.Range.Rows.Count).Row
End Function


'==================== BREAKDOWN TABLES ====================

' Creates/updates a breakdown table and returns its last used row
Private Function UpdateBreakdown(ws As Worksheet, tblName As String, firstColHeader As String, _
                                 periodText As String, dictAll As Object, dictPass As Object, _
                                 anchorRow As Long) As Long
    Dim lo As ListObject
    Dim colCount As Long, colPct As Long
    Dim key As Variant
    Dim body As Range
    Dim rCell As Range
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    ' Create table if missing
    If lo Is Nothing Then
        ws.Range("A" & anchorRow).Value = firstColHeader
        ws.Range("B" & anchorRow).Value = periodText & " - Count"
        ws.Range("C" & anchorRow).Value = periodText & " - Pass %"
        
        Dim initRng As Range
        Set initRng = ws.Range("A" & anchorRow & ":C" & anchorRow)
        Set lo = ws.ListObjects.Add(xlSrcRange, initRng, , xlYes)
        lo.Name = tblName
    End If
    
    ' Ensure period columns exist
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
    
    ' Refresh body range pointer
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
            Set body = lo.ListColumns(1).DataBodyRange
            body.Cells(body.Rows.Count, 1).Value = key
        End If
    Next key
    
    ' If still no body (no keys), return
    On Error Resume Next
    Set body = lo.ListColumns(1).DataBodyRange
    On Error GoTo 0
    If body Is Nothing Then
        UpdateBreakdown = lo.Range.Rows(lo.Range.Rows.Count).Row
        Exit Function
    End If
    
    ' Populate values for this period
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
            If total > 0 Then
                pct = passed / total
            Else
                pct = 0
            End If
            
            rCell.Offset(0, colCount - 1).Value = total
            rCell.Offset(0, colPct - 1).Value = pct
        End If
    Next rCell
    
    ' Formatting
    lo.ListColumns(colCount).DataBodyRange.NumberFormat = "0"
    lo.ListColumns(colPct).DataBodyRange.NumberFormat = "0%"
    
    UpdateBreakdown = lo.Range.Rows(lo.Range.Rows.Count).Row
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
        If StrComp(Trim$(CStr(lc.Name)), Trim$(headerName), vbTextCompare) = 0 Then
            FindHeaderColumn = lc.Index
            Exit Function
        End If
    Next lc
    FindHeaderColumn = 0
End Function