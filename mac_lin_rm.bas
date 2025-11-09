Option Explicit

Sub Build_QA_Dashboard()
    Dim wb As Workbook
    Dim ws As Worksheet
    Dim src As ListObject
    Dim pvtCache As PivotCache
    Dim pvt As PivotTable
    Dim rng As Range
    Dim chartObj As ChartObject
    Dim nextRow As Long
    
    Dim compRng As Range, pfRng As Range
    Dim i As Long
    Dim dt As Date
    Dim minDate As Date, maxDate As Date
    Dim haveDate As Boolean
    Dim periodText As String
    
    Dim dict As Object
    Dim colIncorrect As Long
    Dim key As Variant
    Dim rowOut As Long
    Dim cell As Range
    
    Set wb = ThisWorkbook
    Set src = wb.Worksheets("QA Sample Set").ListObjects("QA_Sam")
    
    '=== Recreate Reporting_Metrics sheet ===
    On Error Resume Next
    Set ws = wb.Worksheets("Reporting_Metrics")
    If Not ws Is Nothing Then
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If
    On Error GoTo 0
    
    Set ws = wb.Worksheets.Add(After:=wb.Worksheets(wb.Worksheets.Count))
    ws.Name = "Reporting_Metrics"
    
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual
    
    '========================================================
    ' 1) Determine period from Completed Date
    '========================================================
    Set compRng = src.ListColumns("Completed Date").DataBodyRange
    Set pfRng = src.ListColumns("Pass/Fail").DataBodyRange
    
    haveDate = False
    For i = 1 To compRng.Rows.Count
        If Trim(pfRng.Cells(i, 1).Value) <> "" Then
            If IsDate(compRng.Cells(i, 1).Value) Then
                dt = CDate(compRng.Cells(i, 1).Value)
                If Not haveDate Then
                    minDate = dt
                    maxDate = dt
                    haveDate = True
                Else
                    If dt < minDate Then minDate = dt
                    If dt > maxDate Then maxDate = dt
                End If
            End If
        End If
    Next i
    
    If haveDate Then
        If Month(minDate) = Month(maxDate) And Year(minDate) = Year(maxDate) Then
            periodText = " - " & Format(minDate, "mmmm yyyy")
        Else
            periodText = " - " & Format(minDate, "mmmm yyyy") & " - " & Format(maxDate, "mmmm yyyy")
        End If
    Else
        periodText = ""
    End If
    
    '========================================================
    ' 2) Summary heading + metrics
    '========================================================
    With ws.Range("A1")
        .Value = "QA Summary Metrics" & periodText
        .Font.Bold = True
        .Font.Size = 14
    End With
    
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
    
    ws.Range("A3:A7").Font.Bold = True
    
    '========================================================
    ' 3) Pivot cache
    '========================================================
    Set pvtCache = wb.PivotCaches.Create( _
        SourceType:=xlDatabase, _
        SourceData:=src.Range)
    
    '========================================================
    ' 4) Overall Pass vs Fail (Donut)
    '========================================================
    nextRow = 10
    ws.Range("A" & nextRow).Value = "Pass vs Fail (Overall)"
    ws.Range("A" & nextRow).Font.Bold = True
    
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, _
        TableDestination:=ws.Range("A" & (nextRow + 1)), _
        TableName:="pvtResultSplit")
    
    With pvt
        .ClearAllFilters
        .PivotFields("Pass/Fail").Orientation = xlRowField
        .AddDataField .PivotFields("Pass/Fail"), "Count of Items", xlCount
    End With
    
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=320, Top:=rng.Top, Width:=260, Height:=200)
    With chartObj.Chart
        .SetSourceData Source:=rng
        .ChartType = xlDoughnut
        .HasTitle = True
        .ChartTitle.Text = "Pass vs Fail"
    End With
    
    '========================================================
    ' 5) Pass/Fail by Reviewer (Bar)
    '========================================================
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Pass/Fail by Reviewer"
    ws.Range("A" & nextRow).Font.Bold = True
    
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, _
        TableDestination:=ws.Range("A" & (nextRow + 1)), _
        TableName:="pvtReviewer")
    
    With pvt
        .ClearAllFilters
        .PivotFields("Reviewer").Orientation = xlRowField
        .PivotFields("Pass/Fail").Orientation = xlColumnField
        
        Dim dfRev As PivotField
        Set dfRev = .AddDataField(.PivotFields("Pass/Fail"), "Count of Items", xlCount)
        dfRev.NumberFormat = "0"
        
        Dim dfRevPct As PivotField
        Set dfRevPct = .AddDataField(.PivotFields("Pass/Fail"), "% of Row", xlCount)
        dfRevPct.Calculation = xlPercentOfRow
        dfRevPct.NumberFormat = "0.0%"
    End With
    
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=320, Top:=rng.Top, Width:=420, Height:=240)
    With chartObj.Chart
        .SetSourceData Source:=rng
        .ChartType = xlColumnClustered
        .HasTitle = True
        .ChartTitle.Text = "Reviewer - Pass/Fail Split"
    End With
    
    '========================================================
    ' 6) Pass/Fail by Tickler Type (Bar)
    '========================================================
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Pass/Fail by Tickler Type"
    ws.Range("A" & nextRow).Font.Bold = True
    
    Set pvt = ws.PivotTables.Add(PivotCache:=pvtCache, _
        TableDestination:=ws.Range("A" & (nextRow + 1)), _
        TableName:="pvtTickler")
    
    With pvt
        .ClearAllFilters
        .PivotFields("Tickler Type").Orientation = xlRowField
        .PivotFields("Pass/Fail").Orientation = xlColumnField
        
        Dim dfT As PivotField
        Set dfT = .AddDataField(.PivotFields("Pass/Fail"), "Count of Items", xlCount)
        dfT.NumberFormat = "0"
        
        Dim dfTPct As PivotField
        Set dfTPct = .AddDataField(.PivotFields("Pass/Fail"), "% of Row", xlCount)
        dfTPct.Calculation = xlPercentOfRow
        dfTPct.NumberFormat = "0.0%"
    End With
    
    Set rng = pvt.TableRange2
    Set chartObj = ws.ChartObjects.Add(Left:=320, Top:=rng.Top, Width:=420, Height:=240)
    With chartObj.Chart
        .SetSourceData Source:=rng
        .ChartType = xlColumnClustered
        .HasTitle = True
        .ChartTitle.Text = "Tickler Type - Pass/Fail Split"
    End With
    
    '========================================================
    ' 7) Incorrect Data Elements Count
    '========================================================
    nextRow = pvt.TableRange2.Row + pvt.TableRange2.Rows.Count + 3
    ws.Range("A" & nextRow).Value = "Incorrect Data Elements Count"
    ws.Range("A" & nextRow).Font.Bold = True
    
    Set dict = CreateObject("Scripting.Dictionary")
    colIncorrect = src.ListColumns("Incorrect Data Elements").Index
    
    For Each cell In src.ListColumns(colIncorrect).DataBodyRange
        If Trim(cell.Value) <> "" Then
            dict(cell.Value) = dict(cell.Value) + 1
        End If
    Next cell
    
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
    
    '========================================================
    ' 8) Cleanup
    '========================================================
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    
    MsgBox "QA Dashboard built successfully in 'Reporting_Metrics'.", vbInformation
End Sub
