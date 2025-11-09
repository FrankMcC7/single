Option Explicit

Sub Update_QA_Metrics()
    Dim wb As Workbook
    Dim wsSrc As Worksheet, wsRep As Worksheet
    Dim lo As ListObject
    Dim dateCol As Range, pfCol As Range, tickCol As Range, revCol As Range, csaCol As Range
    Dim periodText As String
    Dim haveDate As Boolean
    Dim i As Long, dt As Date, minDate As Date, maxDate As Date
    
    Dim colCompletion As Long, colPassFail As Long, colTickler As Long, colReviewer As Long, colCSA As Long
    Dim totalReviewed As Long, totalPass As Long, totalFail As Long
    
    '--- Dictionaries for detail splits ---
    Dim dictTick As Object, dictTickPass As Object
    Dim dictRev As Object, dictRevPass As Object
    Dim dictCSA As Object, dictCSAPass As Object
    Set dictTick = CreateObject("Scripting.Dictionary")
    Set dictTickPass = CreateObject("Scripting.Dictionary")
    Set dictRev = CreateObject("Scripting.Dictionary")
    Set dictRevPass = CreateObject("Scripting.Dictionary")
    Set dictCSA = CreateObject("Scripting.Dictionary")
    Set dictCSAPass = CreateObject("Scripting.Dictionary")
    
    Set wb = ThisWorkbook
    Set wsSrc = wb.Worksheets("QA Sample Set")
    Set lo = wsSrc.ListObjects("QA_Sam")
    
    '=== locate relevant columns ===
    colCompletion = GetListColumnIndex(lo, Array("Completion Date", "Date Reviewed", "Reviewed"))
    colPassFail = GetListColumnIndex(lo, Array("Pass/Fail"))
    colTickler = GetListColumnIndex(lo, Array("Tickler Type"))
    colReviewer = GetListColumnIndex(lo, Array("Reviewer"))
    colCSA = GetListColumnIndex(lo, Array("Offshore CSA (Completed by)", "Offshore CSA"))
    
    If colCompletion = 0 Or colPassFail = 0 Then
        MsgBox "Required columns not found in QA_Sam.", vbCritical
        Exit Sub
    End If
    
    Set dateCol = lo.ListColumns(colCompletion).DataBodyRange
    Set pfCol = lo.ListColumns(colPassFail).DataBodyRange
    Set tickCol = lo.ListColumns(colTickler).DataBodyRange
    Set revCol = lo.ListColumns(colReviewer).DataBodyRange
    Set csaCol = lo.ListColumns(colCSA).DataBodyRange
    
    '=== determine QA period ===
    For i = 1 To dateCol.Rows.Count
        If Trim(pfCol.Cells(i, 1).Value) <> "" And IsDate(dateCol.Cells(i, 1).Value) Then
            dt = CDate(dateCol.Cells(i, 1).Value)
            If Not haveDate Then
                minDate = dt
                maxDate = dt
                haveDate = True
            Else
                If dt < minDate Then minDate = dt
                If dt > maxDate Then maxDate = dt
            End If
        End If
    Next i
    
    If Not haveDate Then
        MsgBox "No valid Completion Dates found with Pass/Fail entries.", vbExclamation
        Exit Sub
    End If
    
    If Month(minDate) = Month(maxDate) And Year(minDate) = Year(maxDate) Then
        periodText = Format(minDate, "mmmm yyyy")
    Else
        periodText = Format(minDate, "mmmm yyyy") & " - " & Format(maxDate, "mmmm yyyy")
    End If
    
    '=== overall counts ===
    For i = 1 To pfCol.Rows.Count
        If Trim(pfCol.Cells(i, 1).Value) <> "" Then
            totalReviewed = totalReviewed + 1
            Select Case LCase(Trim(pfCol.Cells(i, 1).Value))
                Case "pass": totalPass = totalPass + 1
                Case "fail": totalFail = totalFail + 1
            End Select
        End If
    Next i
    
    '=== tickler, reviewer, CSA splits ===
    For i = 1 To pfCol.Rows.Count
        Dim status As String
        status = LCase(Trim(pfCol.Cells(i, 1).Value))
        If status <> "" Then
            ' tickler
            If colTickler > 0 Then
                Dim tKey As String
                tKey = Trim(tickCol.Cells(i, 1).Value)
                If tKey <> "" Then
                    dictTick(tKey) = dictTick(tKey) + 1
                    If status = "pass" Then dictTickPass(tKey) = dictTickPass(tKey) + 1
                End If
            End If
            
            ' reviewer
            If colReviewer > 0 Then
                Dim rKey As String
                rKey = Trim(revCol.Cells(i, 1).Value)
                If rKey <> "" Then
                    dictRev(rKey) = dictRev(rKey) + 1
                    If status = "pass" Then dictRevPass(rKey) = dictRevPass(rKey) + 1
                End If
            End If
            
            ' offshore csa
            If colCSA > 0 Then
                Dim cKey As String
                cKey = Trim(csaCol.Cells(i, 1).Value)
                If cKey <> "" Then
                    dictCSA(cKey) = dictCSA(cKey) + 1
                    If status = "pass" Then dictCSAPass(cKey) = dictCSAPass(cKey) + 1
                End If
            End If
        End If
    Next i
    
    '=== prepare reporting sheet ===
    On Error Resume Next
    Set wsRep = wb.Worksheets("Reporting_Metrics")
    On Error GoTo 0
    If wsRep Is Nothing Then
        Set wsRep = wb.Worksheets.Add
        wsRep.Name = "Reporting_Metrics"
    End If
    
    '=== 1. Summary ===
    AppendSummary wsRep, periodText, totalReviewed, totalPass, totalFail
    
    '=== 2. Tickler ===
    AppendBreakdown wsRep, "Tickler_Type_Metrics", "Tickler Type", periodText, dictTick, dictTickPass
    
    '=== 3. Reviewer ===
    AppendBreakdown wsRep, "Reviewer_Metrics", "Reviewer", periodText, dictRev, dictRevPass
    
    '=== 4. Offshore CSA ===
    AppendBreakdown wsRep, "Offshore_CSA_Metrics", "Offshore CSA (Completed by)", periodText, dictCSA, dictCSAPass
    
    wsRep.Columns.AutoFit
    MsgBox "QA Metrics updated for " & periodText, vbInformation
End Sub

'-----------------------------------------------------------
' APPEND SUMMARY
'-----------------------------------------------------------
Private Sub AppendSummary(ws As Worksheet, periodText As String, totalReviewed As Long, totalPass As Long, totalFail As Long)
    Dim lo As ListObject, rng As Range
    Dim passPct As Double, failPct As Double
    passPct = IIf(totalReviewed = 0, 0, totalPass / totalReviewed)
    failPct = IIf(totalReviewed = 0, 0, totalFail / totalReviewed)
    
    On Error Resume Next
    Set lo = ws.ListObjects("Summary_Stats")
    On Error GoTo 0
    
    If lo Is Nothing Then
        ws.Range("A1").Value = "Metric"
        ws.Range("B1").Value = periodText
        ws.Range("A2").Resize(5, 1).Value = Application.Transpose(Array( _
            "Total QA Reviewed", "Total Pass", "Total Fail", "Pass %", "Fail %"))
        ws.ListObjects.Add xlSrcRange, ws.Range("A1:B6"), , xlYes
        ws.ListObjects(1).Name = "Summary_Stats"
        Set lo = ws.ListObjects("Summary_Stats")
    Else
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText
    End If
    
    Dim colIdx As Long
    colIdx = lo.ListColumns.Count
    With lo.DataBodyRange
        .Cells(1, colIdx - 1 + 1).Value = totalReviewed
        .Cells(2, colIdx - 1 + 1).Value = totalPass
        .Cells(3, colIdx - 1 + 1).Value = totalFail
        .Cells(4, colIdx - 1 + 1).Value = passPct
        .Cells(5, colIdx - 1 + 1).Value = failPct
    End With
    lo.DataBodyRange.Rows(4).Cells(colIdx).NumberFormat = "0.0%"
    lo.DataBodyRange.Rows(5).Cells(colIdx).NumberFormat = "0.0%"
End Sub

'-----------------------------------------------------------
' APPEND BREAKDOWN (Tickler, Reviewer, CSA)
'-----------------------------------------------------------
Private Sub AppendBreakdown(ws As Worksheet, tblName As String, firstColHeader As String, periodText As String, dictAll As Object, dictPass As Object)
    Dim lo As ListObject
    Dim nextCol As Long, key As Variant, found As Range
    Dim colCount As Long, colPct As Long
    Dim anchorRow As Long
    
    On Error Resume Next
    Set lo = ws.ListObjects(tblName)
    On Error GoTo 0
    
    If lo Is Nothing Then
        ' place table a few rows below existing ones dynamically
        anchorRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row + 3
        ws.Range("A" & anchorRow).Value = firstColHeader
        ws.Range("B" & anchorRow).Value = periodText & " - Count"
        ws.Range("C" & anchorRow).Value = periodText & " - Pass %"
        ws.ListObjects.Add xlSrcRange, ws.Range("A" & anchorRow & ":C" & anchorRow), , xlYes
        ws.ListObjects(ws.ListObjects.Count).Name = tblName
        Set lo = ws.ListObjects(tblName)
    Else
        ' add two columns for new period
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count - 1).Value = periodText & " - Count"
        lo.ListColumns.Add
        lo.HeaderRowRange.Cells(1, lo.ListColumns.Count).Value = periodText & " - Pass %"
    End If
    
    colCount = lo.ListColumns.Count - 1
    colPct = lo.ListColumns.Count
    
    ' add missing keys
    For Each key In dictAll.Keys
        Set found = Nothing
        On Error Resume Next
        Set found = lo.ListColumns(1).DataBodyRange.Find(What:=key, LookAt:=xlWhole)
        On Error GoTo 0
        If found Is Nothing Then
            lo.ListRows.Add AlwaysInsert:=True
            lo.ListRows(lo.ListRows.Count).Range.Cells(1, 1).Value = key
        End If
    Next key
    
    ' fill values
    Dim rowCell As Range
    For Each rowCell In lo.ListColumns(1).DataBodyRange
        If dictAll.exists(rowCell.Value) Then
            rowCell.Offset(0, colCount - 1).Value = dictAll(rowCell.Value)
            If dictAll(rowCell.Value) > 0 Then
                rowCell.Offset(0, colPct - 1).Value = dictPass(rowCell.Value) / dictAll(rowCell.Value)
            End If
            rowCell.Offset(0, colPct - 1).NumberFormat = "0.0%"
        End If
    Next rowCell
End Sub

'-----------------------------------------------------------
' FIND COLUMN INDEX
'-----------------------------------------------------------
Private Function GetListColumnIndex(lo As ListObject, names As Variant) As Long
    Dim lc As ListColumn, nm As Variant
    For Each nm In names
        For Each lc In lo.ListColumns
            If StrComp(Trim(lc.Name), Trim(CStr(nm)), vbTextCompare) = 0 Then
                GetListColumnIndex = lc.Index
                Exit Function
            End If
        Next lc
    Next nm
    For Each nm In names
        For Each lc In lo.ListColumns
            If InStr(1, lc.Name, nm, vbTextCompare) > 0 Then
                GetListColumnIndex = lc.Index
                Exit Function
            End If
        Next lc
    Next nm
End Function
