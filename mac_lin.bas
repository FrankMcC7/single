Option Explicit

Sub Build_SrcTbl_And_TicklerCounts()

    Dim wbMain As Workbook
    Dim wbSrc As Workbook
    Dim wsSrc As Worksheet
    Dim wsDest As Worksheet
    Dim wsQA As Worksheet
    Dim tblSrc As ListObject
    Dim tblCount As ListObject
    Dim fDialog As FileDialog
    Dim filePath As String
    
    Dim rngData As Range
    Dim rngVisible As Range
    Dim lastRow As Long, lastCol As Long
    Dim i As Long
    
    Dim ticklerGroupCol As Long
    Dim ticklerTypeCol As Long
    
    Dim dict As Object
    Dim key As Variant
    Dim totalCount As Long
    
    '----------------------------------------------------------------
    ' 0. Set base references
    '----------------------------------------------------------------
    Set wbMain = ThisWorkbook ' Macro runs from MAIN file
    
    On Error Resume Next
    Set wsQA = wbMain.Worksheets("QA Sample Set")
    On Error GoTo 0
    If wsQA Is Nothing Then
        MsgBox "'QA Sample Set' sheet not found in main file.", vbCritical
        Exit Sub
    End If
    
    On Error ResumeNext
    Set tblCount = wsQA.ListObjects("tickler_count")
    On Error GoTo 0
    If tblCount Is Nothing Then
        MsgBox "Table 'tickler_count' not found on 'QA Sample Set' sheet.", vbCritical
        Exit Sub
    End If
    
    '----------------------------------------------------------------
    ' 1. Ask user to select Source File (.xlsx)
    '----------------------------------------------------------------
    Set fDialog = Application.FileDialog(msoFileDialogFilePicker)
    With fDialog
        .Title = "Select Source File (.xlsx)"
        .Filters.Clear
        .Filters.Add "Excel Files", "*.xlsx"
        .AllowMultiSelect = False
        If .Show <> -1 Then
            Exit Sub ' User cancelled
        End If
        filePath = .SelectedItems(1)
    End With
    
    '----------------------------------------------------------------
    ' 2. Open Source File (Read-Only) & identify data range
    '----------------------------------------------------------------
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual
    
    Set wbSrc = Workbooks.Open(Filename:=filePath, ReadOnly:=True)
    
    'Assumption: data is on the FIRST sheet of source file
    Set wsSrc = wbSrc.Sheets(1)
    
    With wsSrc
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        If lastRow < 2 Or lastCol < 1 Then
            MsgBox "No usable data found in source file.", vbCritical
            GoTo CleanExit
        End If
        Set rngData = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
    End With
    
    '----------------------------------------------------------------
    ' 3. Remove/Exclude rows with Tickler Group = "AER" and copy to main
    '    into new sheet 'Source file' as table 'Src_tbl'
    '----------------------------------------------------------------
    'Find "Tickler Group" column in source header row
    ticklerGroupCol = 0
    For i = 1 To rngData.Columns.Count
        If LCase(Trim(rngData.Cells(1, i).Value)) = LCase("Tickler Group") Then
            ticklerGroupCol = i
            Exit For
        End If
    Next i
    
    'Delete existing "Source file" sheet in main (if any)
    On Error ResumeNext
    Application.DisplayAlerts = False
    wbMain.Worksheets("Source file").Delete
    Application.DisplayAlerts = True
    On Error GoTo 0
    
    'Create fresh "Source file" sheet in main
    Set wsDest = wbMain.Worksheets.Add(After:=wbMain.Sheets(wbMain.Sheets.Count))
    wsDest.Name = "Source file"
    
    'If Tickler Group column found, filter out "AER"
    If ticklerGroupCol > 0 Then
        rngData.AutoFilter Field:=ticklerGroupCol, Criteria1:="<>" & "AER"
        On Error ResumeNext
        Set rngVisible = rngData.SpecialCells(xlCellTypeVisible)
        On Error GoTo 0
        
        If rngVisible Is Nothing Then
            MsgBox "All rows filtered out by Tickler Group = 'AER'. No data to copy.", vbExclamation
            wsDest.Delete
            GoTo CleanExit
        End If
        
        rngVisible.Copy Destination:=wsDest.Range("A1")
        wsSrc.AutoFilterMode = False
        
    Else
        'If no Tickler Group col, just copy entire dataset
        rngData.Copy Destination:=wsDest.Range("A1")
    End If
    
    'Close Source File without changes
    wbSrc.Close SaveChanges:=False
    
    'Create table Src_tbl on "Source file"
    With wsDest
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        If lastRow < 2 Then
            MsgBox "No data copied into 'Source file' sheet.", vbCritical
            GoTo CleanExit
        End If
        
        Set rngData = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
    End With
    
    'Remove any existing table named Src_tbl (if from earlier runs)
    On Error ResumeNext
    Dim lo As ListObject
    For Each lo In wsDest.ListObjects
        If lo.Name = "Src_tbl" Then
            lo.Unlist
            Exit For
        End If
    Next lo
    On Error GoTo 0
    
    Set tblSrc = wsDest.ListObjects.Add(SourceType:=xlSrcRange, _
                                        Source:=rngData, _
                                        XlListObjectHasHeaders:=xlYes)
    tblSrc.Name = "Src_tbl"
    
    '----------------------------------------------------------------
    ' 4. Populate tickler_count table on 'QA Sample Set'
    '    - Unique Tickler Type from Src_tbl[Tickler Type]
    '    - Source Count = count per Tickler Type
    '    - % of Total = count / total ticklers (rows in Src_tbl)
    '----------------------------------------------------------------
    
    'Find Tickler Type column in Src_tbl
    ticklerTypeCol = 0
    For i = 1 To tblSrc.ListColumns.Count
        If LCase(Trim(tblSrc.ListColumns(i).Name)) = LCase("Tickler Type") Then
            ticklerTypeCol = i
            Exit For
        End If
    Next i
    
    If ticklerTypeCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If
    
    'Clear existing data rows in tickler_count
    If tblCount.ListRows.Count > 0 Then
        tblCount.DataBodyRange.Delete
    End If
    
    'Build dictionary of Tickler Type counts
    Set dict = CreateObject("Scripting.Dictionary")
    dict.CompareMode = vbTextCompare
    
    Dim tick As String
    Dim r As Long
    
    totalCount = 0
    With tblSrc.DataBodyRange
        For r = 1 To .Rows.Count
            tick = Trim(CStr(.Cells(r, ticklerTypeCol).Value))
            If tick <> "" Then
                totalCount = totalCount + 1
                If dict.Exists(tick) Then
                    dict(tick) = dict(tick) + 1
                Else
                    dict.Add tick, 1
                End If
            End If
        Next r
    End With
    
    'Populate tickler_count rows
    If dict.Count = 0 Or totalCount = 0 Then
        MsgBox "No Tickler Type data found in Src_tbl to populate tickler_count.", vbExclamation
        GoTo CleanExit
    End If
    
    For Each key In dict.Keys
        With tblCount.ListRows.Add
            'Assumes:
            ' Col 1 = Tickler Type
            ' Col 2 = Source Count
            ' Col 3 = % of Total
            .Range(1, 1).Value = key
            .Range(1, 2).Value = dict(key)
            .Range(1, 3).Value = dict(key) / totalCount
        End With
    Next key
    
    'Format % column
    With tblCount.ListColumns(3).DataBodyRange
        .NumberFormat = "0.00%"
    End With
    
    '----------------------------------------------------------------
    ' Done (end of first macro)
    '----------------------------------------------------------------
    MsgBox "Step 1 complete:" & vbCrLf & _
           "- Source file loaded & cleaned (AER removed)." & vbCrLf & _
           "- Src_tbl created on 'Source file' sheet." & vbCrLf & _
           "- tickler_count table populated.", vbInformation

CleanExit:
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Application.Calculation = xlCalculationAutomatic

End Sub
