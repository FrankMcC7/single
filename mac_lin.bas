Option Explicit

Sub Build_SrcTbl_And_TicklerCounts()

    '========================================================
    ' 1) Runs from MAIN workbook.
    ' 2) Prompts user to select Source File (.xlsx).
    ' 3) Copies data excluding Tickler Group = "AER"
    '    into new sheet "Source file" as table Src_tbl.
    ' 4) Populates tickler_count table on "QA Sample Set":
    '       - Tickler Type (unique from Src_tbl[Tickler Type])
    '       - Source Count (frequency)
    '       - % of Total (count / total non-blank ticklers)
    '    Uses existing tickler_count rows; does NOT resize table
    '    to avoid conflict with table below.
    '========================================================

    Dim wbMain As Workbook
    Dim wbSrc As Workbook
    Dim wsSrc As Worksheet
    Dim wsDest As Worksheet
    Dim wsQA As Worksheet
    
    Dim tblSrc As ListObject
    Dim tblCount As ListObject
    Dim lo As ListObject
    
    Dim fDialog As FileDialog
    Dim filePath As String
    
    Dim rngData As Range
    Dim rngVisible As Range
    Dim bodyRange As Range
    
    Dim lastRow As Long
    Dim lastCol As Long
    Dim i As Long
    Dim r As Long
    
    Dim ticklerGroupCol As Long
    Dim ticklerTypeCol As Long
    
    Dim dict As Object
    Dim key As Variant
    Dim totalCount As Long
    Dim tick As String
    
    Dim maxRows As Long
    Dim rowIdx As Long

    On Error GoTo ErrHandler
    
    Set wbMain = ThisWorkbook

'--------------------------------------------------------
' Check and delete existing "Source file" sheet if present
'--------------------------------------------------------
Dim ws As Worksheet

On Error Resume Next
Set ws = wbMain.Worksheets("Source file")
On Error GoTo 0

If Not ws Is Nothing Then
    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True
End If

    '--------------------------------------------------------
    ' Validate "QA Sample Set" sheet and tickler_count table
    '--------------------------------------------------------
    On Error Resume Next
    Set wsQA = wbMain.Worksheets("QA Sample Set")
    If Not wsQA Is Nothing Then
        Set tblCount = wsQA.ListObjects("tickler_count")
    End If
    On Error GoTo ErrHandler
    
    If wsQA Is Nothing Then
        MsgBox "'QA Sample Set' sheet not found in main file.", vbCritical
        GoTo CleanExit
    End If
    
    If tblCount Is Nothing Then
        MsgBox "Table 'tickler_count' not found on 'QA Sample Set' sheet.", vbCritical
        GoTo CleanExit
    End If

    '--------------------------------------------------------
    ' Prompt user for Source File (.xlsx)
    '--------------------------------------------------------
    Set fDialog = Application.FileDialog(msoFileDialogFilePicker)
    With fDialog
        .Title = "Select Source File (.xlsx)"
        .Filters.Clear
        .Filters.Add "Excel Files", "*.xlsx"
        .AllowMultiSelect = False
        If .Show <> -1 Then
            GoTo CleanExit ' User cancelled
        End If
        filePath = .SelectedItems(1)
    End With

    '--------------------------------------------------------
    ' Performance settings
    '--------------------------------------------------------
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual
    Application.DisplayAlerts = False

    '--------------------------------------------------------
    ' Open Source File (Read-Only) & get data range
    '--------------------------------------------------------
    Set wbSrc = Workbooks.Open(Filename:=filePath, ReadOnly:=True)
    Set wsSrc = wbSrc.Sheets(1) ' Assuming first sheet has data
    
    With wsSrc
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        If lastRow < 2 Or lastCol < 1 Then
            MsgBox "No usable data found in source file.", vbCritical
            GoTo CleanExit
        End If
        Set rngData = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
    End With

    '--------------------------------------------------------
    ' Recreate "Source file" sheet in MAIN workbook
    '--------------------------------------------------------
    On Error Resume Next
    wbMain.Worksheets("Source file").Delete
    On Error GoTo ErrHandler
    
    Set wsDest = wbMain.Worksheets.Add(After:=wbMain.Sheets(wbMain.Sheets.Count))
    wsDest.Name = "Source file"

    '--------------------------------------------------------
    ' Find "Tickler Group" column and filter out AER
    '--------------------------------------------------------
    ticklerGroupCol = 0
    For i = 1 To rngData.Columns.Count
        If LCase$(Trim$(rngData.Cells(1, i).Value)) = "tickler group" Then
            ticklerGroupCol = i
            Exit For
        End If
    Next i

    If ticklerGroupCol > 0 Then
        rngData.AutoFilter Field:=ticklerGroupCol, Criteria1:="<>" & "AER"
        
        On Error Resume Next
        Set rngVisible = rngData.SpecialCells(xlCellTypeVisible)
        On Error GoTo ErrHandler
        
        If rngVisible Is Nothing Then
            MsgBox "All rows excluded by Tickler Group = 'AER'. No data to copy.", vbExclamation
            GoTo CleanExit
        End If
        
        rngVisible.Copy Destination:=wsDest.Range("A1")
        wsSrc.AutoFilterMode = False
    Else
        ' If "Tickler Group" not found, copy all data
        rngData.Copy Destination:=wsDest.Range("A1")
    End If

    '--------------------------------------------------------
    ' Close Source File without saving
    '--------------------------------------------------------
    wbSrc.Close SaveChanges:=False
    Set wbSrc = Nothing
    Set wsSrc = Nothing

    '--------------------------------------------------------
    ' Create Src_tbl on "Source file"
    '--------------------------------------------------------
    With wsDest
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        If lastRow < 2 Or lastCol < 1 Then
            MsgBox "No data copied into 'Source file' sheet.", vbCritical
            GoTo CleanExit
        End If
        Set rngData = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
    End With

    ' Remove any existing Src_tbl (safety)
    On Error Resume Next
    For Each lo In wsDest.ListObjects
        If lo.Name = "Src_tbl" Then
            lo.Unlist
            Exit For
        End If
    Next lo
    On Error GoTo ErrHandler

    Set tblSrc = wsDest.ListObjects.Add( _
                    SourceType:=xlSrcRange, _
                    Source:=rngData, _
                    XlListObjectHasHeaders:=xlYes)
    tblSrc.Name = "Src_tbl"

    '--------------------------------------------------------
    ' Locate "Tickler Type" column in Src_tbl
    '--------------------------------------------------------
    ticklerTypeCol = 0
    For i = 1 To tblSrc.ListColumns.Count
        If LCase$(Trim$(tblSrc.ListColumns(i).Name)) = "tickler type" Then
            ticklerTypeCol = i
            Exit For
        End If
    Next i

    If ticklerTypeCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If

    '--------------------------------------------------------
    ' Prepare tickler_count table body (no resizing)
    '--------------------------------------------------------
    If tblCount.ListRows.Count = 0 Then
        tblCount.ListRows.Add
    End If

    Set bodyRange = tblCount.DataBodyRange
    bodyRange.ClearContents
    maxRows = bodyRange.Rows.Count

    '--------------------------------------------------------
    ' Build dictionary of Tickler Type counts
    '--------------------------------------------------------
    Set dict = CreateObject("Scripting.Dictionary")
    dict.CompareMode = vbTextCompare

    totalCount = 0
    With tblSrc.DataBodyRange
        For r = 1 To .Rows.Count
            tick = Trim$(CStr(.Cells(r, ticklerTypeCol).Value))
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

    If dict.Count = 0 Or totalCount = 0 Then
        MsgBox "No Tickler Type data found in Src_tbl to populate tickler_count.", vbExclamation
        GoTo CleanExit
    End If

    '--------------------------------------------------------
    ' Check capacity vs existing tickler_count rows
    '--------------------------------------------------------
    If dict.Count > maxRows Then
        MsgBox "There are " & dict.Count & " unique Tickler Types," & vbCrLf & _
               "but only " & maxRows & " rows available in 'tickler_count' table." & vbCrLf & _
               "Add more rows to tickler_count (above the next table) and rerun.", _
               vbCritical
        GoTo CleanExit
    End If

    '--------------------------------------------------------
    ' Populate tickler_count within existing rows
    '   Col1 = Tickler Type
    '   Col2 = Source Count
    '   Col3 = % of Total
    '--------------------------------------------------------
    rowIdx = 1
    For Each key In dict.Keys
        With bodyRange.Rows(rowIdx)
            .Cells(1, 1).Value = key
            .Cells(1, 2).Value = dict(key)
            .Cells(1, 3).Value = dict(key) / totalCount
        End With
        rowIdx = rowIdx + 1
    Next key

    ' Format % column
    With tblCount.ListColumns(3).DataBodyRange
        .NumberFormat = "0.00%"
    End With

    '--------------------------------------------------------
    ' Done
    '--------------------------------------------------------
    MsgBox "Macro completed:" & vbCrLf & _
           "- 'Source file' sheet created with Src_tbl (AER removed)." & vbCrLf & _
           "- 'tickler_count' table populated within pre-defined rows.", vbInformation

CleanExit:
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Application.Calculation = xlCalculationAutomatic
    Application.DisplayAlerts = True
    Exit Sub

ErrHandler:
    MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical, "Build_SrcTbl_And_TicklerCounts"
    Resume CleanExit

End Sub
