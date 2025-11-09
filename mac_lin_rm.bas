Option Explicit

Sub Build_QA_Dashboard()
    Dim ws As Worksheet, src As ListObject
    Dim pvtCache As PivotCache
    Dim pvt As PivotTable
    Dim rng As Range
    Dim chartObj As ChartObject
    Dim nextRow As Long
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    '--- references
    Set src = ThisWorkbook.Worksheets("QA Sample Set").ListObjects("QA_Sam")
    
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets("Reporting_Metrics")
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = "Reporting_Metrics"
    End If
    On Error GoTo 0
    
    '--- clear sheet
    ws.Cells.Clear
    
    '=== SUMMARY METRICS ===
    ws.Range("A1").Value = "QA Summary Metrics"
    ws.Range("A1").Font.Bold = True
    ws.Range("A1").Font.Size = 14
    
    ws.Range("A3").Value = "Total QA Reviewed"
    ws.Range("B3").Formula = "=COUNTIF(QA_Sam[Pass/Fail],""<>"")"
    
    ws.Range("A4").Value = "Total Pass"
    ws.Range("B4").Formula = "=COUNTIF(QA_Sam[Pass/Fail],""Pass"")"
    
    ws.Range("A5").Value = "Total Fail"
    ws.Range("B5").Formula = "=COUNTIF(QA_Sam[Pass/Fail],""Fail"")"
    
    ws.Range("A6").Value = "Pass %"
    ws.Range("B6").Formula = "=IF(B3=0,"""",B4/B3)"
    ws.Range("B6").NumberFormat = "0.0%"
    
    ws.Range("A7").Value = "Fail %"
    ws.Range("B7").Formula = "=IF(B3=0,"""",B5/B3)"
    ws.Range("B7").NumberFormat = "0.0%"
    
    '=== PIVOT CACHE ===
    Set pvtCache = ThisWorkbook.PivotCaches.Create( _
        SourceType:=xlDatabase, _
        SourceData:=src.Range)
    
    nextRow = 10
    
    '=== 1. PASS/FAIL OVERALL ===
    ws.Range("A" & nextRow).Value = "Pass/Fail Split"
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, TableDestination:=ws.Range("A" & (nextRow + 1)), TableName:="pvtResultSplit")
    With pvt
        .PivotFields("Pass/Fail").Orientation = xlRowField
        .AddDataField .PivotFields("Pass/Fail"), "Count", xlCount
    End With
    
    ' Donut chart
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=300, Top:=rng.Top, Width:=250, Height:=200)
    With chartObj.Chart
        .SetSourceData Source:=rng
        .ChartType = xlDoughnut
        .HasTitle = True
        .ChartTitle.Text = "Pass vs Fail"
    End With
    
    '=== 2. PASS/FAIL BY REVIEWER ===
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Pass/Fail by Reviewer"
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, TableDestination:=ws.Range("A" & (nextRow + 1)), TableName:="pvtReviewer")
    With pvt
        .PivotFields("Reviewer").Orientation = xlRowField
        .PivotFields("Pass/Fail").Orientation = xlColumnField
        .AddDataField .PivotFields("Pass/Fail"), "Count", xlCount
        .RowAxisLayout xlTabularRow
    End With
    
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=300, Top:=rng.Top, Width:=400, Height:=250)
    With chartObj.Chart
        .SetSourceData rng
        .ChartType = xlColumnClustered
        .HasTitle = True
        .ChartTitle.Text = "Reviewer Pass vs Fail"
    End With
    
    '=== 3. PASS/FAIL BY TICKLER TYPE ===
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Pass/Fail by Tickler Type"
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, TableDestination:=ws.Range("A" & (nextRow + 1)), TableName:="pvtTickler")
    With pvt
        .PivotFields("Tickler Type").Orientation = xlRowField
        .PivotFields("Pass/Fail").Orientation = xlColumnField
        .AddDataField .PivotFields("Pass/Fail"), "Count", xlCount
        .RowAxisLayout xlTabularRow
    End With
    
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=300, Top:=rng.Top, Width:=400, Height:=250)
    With chartObj.Chart
        .SetSourceData rng
        .ChartType = xlColumnClustered
        .HasTitle = True
        .ChartTitle.Text = "Tickler Type Pass vs Fail"
    End With
    
    '=== 4. INCORRECT DATA ELEMENTS COUNT ===
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Incorrect Data Elements Count"
    ws.Range("A" & nextRow).Font.Bold = True
    
    Dim dict As Object, c As Range, key As Variant
    Dim colIndex As Long, rowOut As Long
    Set dict = CreateObject("Scripting.Dictionary")
    
    ' find column
    colIndex = src.ListColumns("Incorrect Data Elements").Index
    
    For Each c In src.ListColumns(colIndex).DataBodyRange
        If Trim(c.Value) <> "" Then
            dict(c.Value) = dict(c.Value) + 1
        End If
    Next c
    
    rowOut = nextRow + 1
    ws.Range("A" & rowOut).Value = "Incorrect Data Element"
    ws.Range("B" & rowOut).Value = "Count"
    ws.Range("A" & rowOut & ":B" & rowOut).Font.Bold = True
    
    For Each key In dict.keys
        rowOut = rowOut + 1
        ws.Range("A" & rowOut).Value = key
        ws.Range("B" & rowOut).Value = dict(key)
    Next key
    
    ws.Columns.AutoFit
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True
    
    MsgBox "QA Dashboard built successfully in 'Reporting_Metrics'.", vbInformation
End Sub
