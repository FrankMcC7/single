Option Explicit

'=========================================================================================
' README — Macro 1: Build_SrcTbl_And_TicklerCounts
'
' WHAT THIS DOES
' 1) Runs from the MAIN workbook (the one containing sheets "QA_Parameters", "QA Sample Set", "Keys").
' 2) If a sheet named "Source file" already exists in MAIN, deletes it (fresh run each time).
' 3) Prompts you to pick a Source File (.xlsx). Opens it Read-Only.
' 4) Copies all rows into a new MAIN sheet "Source file" with these EXCLUSIONS applied:
'       - If column "Tickler Group" exists: exclude rows where Tickler Group = "AER".
'       - If column "Tickler Type" exists: exclude rows where Tickler Type is:
'             "Monitoring"
'             "CAM Annual Review"
'             "Condition to Approva"
'             "Monitoring Review"
'       - If column "Tickler Status" exists: exclude rows where Tickler Status = "Waived".
' 5) Converts the resulting data on "Source file" to a table named "Src_tbl".
' 6) Populates the table "tickler_count" (on sheet "QA_Parameters"):
'       - Column "Tickler Type": unique values from Src_tbl[Tickler Type]
'       - Column "Source Count": frequency of each Tickler Type
'       - Column "% of Total"  : Source Count / total non-blank Tickler Type rows (as %)
' 7) Adds a **Grand Total %** (SUM of "% of Total") in the sheet, just below the
'    "tickler_count" table, so users can see if they have changed the individual % values.
'
' LAYOUT & REQUIREMENTS (assumed)
' - MAIN workbook has sheet "QA_Parameters" with table "tickler_count"
'   Required headers: "Tickler Type", "Source Count", "% of Total"
'   NOTE: This macro DOES NOT resize "tickler_count" (to avoid pushing into a table below).
'         Pre-allocate enough data rows. The macro writes from the first row down and
'         clears remaining rows if there are fewer unique values than preallocated rows.
'         The "Grand Total %" cell is placed OUTSIDE the table (one row below).
'
' - Source file (you pick) has a header row.
'   The macro *safely ignores* missing columns:
'       • If "Tickler Group" is missing, no AER exclusion happens.
'       • If "Tickler Type" is missing, no type-based omission happens.
'       • If "Tickler Status" is missing, no status-based omission happens.
'
' COMMON ISSUES AVOIDED
' - Table-on-table collision: we never add ListRows to "tickler_count".
' - "Grand Total %" is outside the table, so we do not resize it.
' - Source file stays unchanged (opened Read-Only, then closed without saving).
'
' HOW TO RUN
' - Put this code in a standard module in the MAIN workbook.
' - Ensure sheet "QA_Parameters" has table "tickler_count" with enough rows.
' - Run: Build_SrcTbl_And_TicklerCounts
'=========================================================================================

Sub Build_SrcTbl_And_TicklerCounts()

    '----------------------------
    ' Declarations
    '----------------------------
    Dim wbMain As Workbook, wbSrc As Workbook
    Dim wsQA As Worksheet, wsDest As Worksheet, wsSrc As Worksheet
    Dim tblSrc As ListObject, tblCount As ListObject, lo As ListObject
    Dim fDialog As FileDialog
    Dim filePath As String
    
    Dim rngData As Range, rngVisible As Range, rngAll As Range
    Dim lastRow As Long, lastCol As Long
    Dim i As Long, r As Long
    
    Dim ticklerGroupCol As Long, ticklerTypeCol As Long
    Dim ticklerTypeColDest As Long, ticklerStatusColDest As Long
    Dim totalCount As Long
    Dim dict As Object ' Scripting.Dictionary
    Dim key As Variant
    
    Dim bodyCount As Range
    Dim maxRows As Long, rowIdx As Long
    
    Dim valType As String, valStatus As String
    Dim omitRow As Boolean
    
    Dim grandTotalCell As Range
    Dim grandTotalLabelCell As Range
    
    On Error GoTo ErrHandler
    Set wbMain = ThisWorkbook

    '--------------------------------------------------------
    ' Delete existing "Source file" sheet if present (fresh run)
    '--------------------------------------------------------
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = wbMain.Worksheets("Source file")
    On Error GoTo ErrHandler
    If Not ws Is Nothing Then
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If
    
    '--------------------------------------------------------
    ' Validate "QA_Parameters" and "tickler_count"
    '--------------------------------------------------------
    Set wsQA = wbMain.Worksheets("QA_Parameters")
    Set tblCount = wsQA.ListObjects("tickler_count")
    
    ' Grab body range (ensure at least one row exists)
    If tblCount.ListRows.Count = 0 Then
        ' add one blank row so DataBodyRange is not Nothing
        tblCount.ListRows.Add
    End If
    Set bodyCount = tblCount.DataBodyRange
    maxRows = bodyCount.Rows.Count
    
    '--------------------------------------------------------
    ' Ask user for Source File (.xlsx)
    '--------------------------------------------------------
    Set fDialog = Application.FileDialog(msoFileDialogFilePicker)
    With fDialog
        .Title = "Select Source File (.xlsx)"
        .Filters.Clear
        .Filters.Add "Excel Files", "*.xlsx"
        .AllowMultiSelect = False
        If .Show <> -1 Then GoTo CleanExit ' cancelled
        filePath = .SelectedItems(1)
    End With
    
    '--------------------------------------------------------
    ' Speed up
    '--------------------------------------------------------
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual
    
    '--------------------------------------------------------
    ' Open source (ReadOnly) & get used range
    '--------------------------------------------------------
    Set wbSrc = Workbooks.Open(Filename:=filePath, ReadOnly:=True)
    Set wsSrc = wbSrc.Sheets(1) ' assume first sheet
    
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
    ' Create fresh "Source file" sheet in MAIN
    '--------------------------------------------------------
    Set wsDest = wbMain.Worksheets.Add(After:=wbMain.Sheets(wbMain.Sheets.Count))
    wsDest.Name = "Source file"
    
    '--------------------------------------------------------
    ' Exclude Tickler Group = "AER" if that column exists (at source)
    '   NOTE: Additional Tickler Type / Status exclusions are applied
    '         AFTER copying, on the "Source file" sheet.
    '--------------------------------------------------------
    ticklerGroupCol = FindHeaderColumnIndex(rngData, "Tickler Group")
    
    If ticklerGroupCol > 0 Then
        rngData.AutoFilter Field:=ticklerGroupCol, Criteria1:="<>" & "AER"
        On Error Resume Next
        Set rngVisible = rngData.SpecialCells(xlCellTypeVisible)
        On Error GoTo ErrHandler
        
        If rngVisible Is Nothing Then
            MsgBox "All rows excluded by Tickler Group = 'AER'. No data to copy.", vbExclamation
            GoTo CloseSourceAndExit
        End If
        
        rngVisible.Copy Destination:=wsDest.Range("A1")
        wsSrc.AutoFilterMode = False
    Else
        ' No Tickler Group column → copy all
        rngData.Copy Destination:=wsDest.Range("A1")
    End If
    
CloseSourceAndExit:
    ' Close source file without saving (leave it unchanged)
    wbSrc.Close SaveChanges:=False
    Set wbSrc = Nothing
    Set wsSrc = Nothing
    
    '--------------------------------------------------------
    ' Apply extra exclusions on the copied data in "Source file":
    '   - Omit specific Tickler Type values
    '   - Omit Tickler Status = "Waived"
    '--------------------------------------------------------
    With wsDest
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        
        If lastRow >= 2 And lastCol >= 1 Then
            Set rngAll = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
            
            ticklerTypeColDest = FindHeaderColumnIndex(rngAll, "Tickler Type")
            ticklerStatusColDest = FindHeaderColumnIndex(rngAll, "Tickler Status")
            
            If ticklerTypeColDest > 0 Or ticklerStatusColDest > 0 Then
                For r = lastRow To 2 Step -1 ' bottom-up, skip header
                    valType = ""
                    valStatus = ""
                    omitRow = False
                    
                    If ticklerTypeColDest > 0 Then
                        valType = Trim$(CStr(.Cells(r, ticklerTypeColDest).Value))
                        Select Case LCase$(valType)
                            Case "monitoring", _
                                 "cam annual review", _
                                 "condition to approva", _
                                 "monitoring review"
                                omitRow = True
                        End Select
                    End If
                    
                    If Not omitRow And ticklerStatusColDest > 0 Then
                        valStatus = Trim$(CStr(.Cells(r, ticklerStatusColDest).Value))
                        If LCase$(valStatus) = "waived" Then
                            omitRow = True
                        End If
                    End If
                    
                    If omitRow Then
                        .Rows(r).Delete
                    End If
                Next r
            End If
        End If
    End With
    
    '--------------------------------------------------------
    ' Make the cleaned range a table: Src_tbl
    '--------------------------------------------------------
    With wsDest
        lastRow = .Cells(.Rows.Count, 1).End(xlUp).Row
        lastCol = .Cells(1, .Columns.Count).End(xlToLeft).Column
        If lastRow < 2 Then
            MsgBox "No data remaining in 'Source file' sheet after exclusions.", vbCritical
            GoTo CleanExit
        End If
        Set rngData = .Range(.Cells(1, 1), .Cells(lastRow, lastCol))
    End With
    
    ' Remove any existing "Src_tbl" on this sheet (safety)
    On Error Resume Next
    For Each lo In wsDest.ListObjects
        If lo.Name = "Src_tbl" Then
            lo.Unlist
            Exit For
        End If
    Next lo
    On Error GoTo ErrHandler
    
    Set tblSrc = wsDest.ListObjects.Add(SourceType:=xlSrcRange, _
                                        Source:=rngData, _
                                        XlListObjectHasHeaders:=xlYes)
    tblSrc.Name = "Src_tbl"
    
    '--------------------------------------------------------
    ' Build counts for unique Tickler Type
    '--------------------------------------------------------
    ticklerTypeCol = FindListObjectColumnIndex(tblSrc, "Tickler Type")
    If ticklerTypeCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If
    
    Set dict = CreateObject("Scripting.Dictionary")
    dict.CompareMode = vbTextCompare
    
    totalCount = 0
    With tblSrc.DataBodyRange
        For r = 1 To .Rows.Count
            Dim tick As String
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
    
    '--------------------------------------------------------
    ' Populate tickler_count WITHOUT resizing the table
    '   Col1 = Tickler Type
    '   Col2 = Source Count
    '   Col3 = % of Total
    '--------------------------------------------------------
    If totalCount = 0 Then
        ' Clear any existing numbers
        bodyCount.ClearContents
        MsgBox "No non-blank 'Tickler Type' values found in Src_tbl.", vbExclamation
        GoTo CleanExit
    End If
    
    ' Clear body FIRST (keep headers and row count)
    bodyCount.ClearContents
    
    rowIdx = 1
    For Each key In dict.Keys
        If rowIdx > maxRows Then
            MsgBox "Unique Tickler Types (" & dict.Count & _
                   ") exceed available rows (" & maxRows & _
                   ") in 'tickler_count'. Add more rows and rerun.", vbCritical
            Exit For
        End If
        
        bodyCount.Cells(rowIdx, 1).Value = CStr(key)          ' Tickler Type
        bodyCount.Cells(rowIdx, 2).Value = CLng(dict(key))    ' Source Count
        bodyCount.Cells(rowIdx, 3).Value = CDbl(dict(key)) / totalCount ' % of Total
        rowIdx = rowIdx + 1
    Next key
    
    ' Format % column
    With tblCount.ListColumns(3).DataBodyRange
        .NumberFormat = "0.00%"
    End With
    
    '--------------------------------------------------------
    ' Add GRAND TOTAL % just below the table (outside tickler_count)
    '   - Label in column 2: "Grand Total %"
    '   - Formula in column 3: =SUM(<% of Total column in DataBodyRange>)
    '   This remains outside the ListObject so the table does not resize.
    '--------------------------------------------------------
    Set grandTotalCell = bodyCount.Cells(maxRows, 3).Offset(1, 0)       ' one row below last body row
    Set grandTotalLabelCell = grandTotalCell.Offset(0, -1)              ' column just to the left
    
    grandTotalLabelCell.Value = "Grand Total %"
    grandTotalCell.Formula = "=SUM(" & bodyCount.Columns(3).Address(True, True) & ")"
    grandTotalCell.NumberFormat = "0.00%"
    
    ' If fewer unique types than preallocated rows → rest already blank
    
    '--------------------------------------------------------
    ' Done
    '--------------------------------------------------------
    MsgBox "Step 1 complete:" & vbCrLf & _
           "- Old 'Source file' (if any) removed and rebuilt." & vbCrLf & _
           "- Src_tbl created (AER / banned Tickler Types / Waived status excluded)." & vbCrLf & _
           "- tickler_count filled on 'QA_Parameters' without resizing." & vbCrLf & _
           "- Grand Total % added below tickler_count.", vbInformation

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

'-----------------------------------------------------------------------------------------
' Helper: Find a header column by name in a 2D Range (first row is headers)
' Returns 0 if not found
'-----------------------------------------------------------------------------------------
Private Function FindHeaderColumnIndex(ByVal rng As Range, ByVal headerName As String) As Long
    Dim i As Long, target As String
    target = LCase$(Trim$(headerName))
    For i = 1 To rng.Columns.Count
        If LCase$(Trim$(CStr(rng.Cells(1, i).Value))) = target Then
            FindHeaderColumnIndex = i
            Exit Function
        End If
    Next i
    FindHeaderColumnIndex = 0
End Function

'-----------------------------------------------------------------------------------------
' Helper: Find a ListObject column index by header name
' Returns 0 if not found
'-----------------------------------------------------------------------------------------
Private Function FindListObjectColumnIndex(ByVal tbl As ListObject, ByVal headerName As String) As Long
    Dim i As Long, target As String
    target = LCase$(Trim$(headerName))
    For i = 1 To tbl.ListColumns.Count
        If LCase$(Trim$(tbl.ListColumns(i).Name)) = target Then
            FindListObjectColumnIndex = i
            Exit Function
        End If
    Next i
    FindListObjectColumnIndex = 0
End Function
