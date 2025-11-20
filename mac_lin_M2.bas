Option Explicit

'=========================================================================================
' README — Macro 2: Build_QA_Sample_Set (UPDATED LAYOUT & FILTER RESET)
'
' PURPOSE
'   Build a randomized QA sample set from Src_tbl into QA_Sam.
'
' WHAT IT DOES
'   1) Ensures QA_Sam table is fully UNFILTERED (clears any filters from previous macros).
'   2) Clears only the contents of QA_Sam (keeps table structure/validations).
'   3) In QA_Parameters!tickler_count, sets [Sample Set Count] = ROUNDUP([% of Total] * F3, 0).
'   4) Reads column mappings from Keys!col_key (or Keys!col_keys):
'        - QA_Sam_col  → column header in QA_Sam
'        - Src_tbl_col → corresponding source column header in Src_tbl
'   5) For each Tickler Type with Sample Set Count > 0:
'        - Randomly select N rows from Src_tbl where N = Sample Set Count.
'        - Append those rows to QA_Sam (ListRows.Add) using the mapping.
'   6) Deletes any fully blank rows within QA_Sam so the table starts clean from the top.
'
' UPDATED LAYOUT (per your new guidance)
'   - Sample Set Size:      'QA_Parameters'!F3
'   - tickler_count table:  on sheet 'QA_Parameters'
'   - QA_Sam table:         on sheet 'QA Sample Set'
'
' ASSUMPTIONS / REQUIREMENTS
'   - MAIN workbook contains:
'       • Sheet "Source file" with table "Src_tbl"
'       • Sheet "QA_Parameters" with table "tickler_count"
'           Required headers: "Tickler Type", "% of Total", "Sample Set Count"
'       • Sheet "QA Sample Set" with table "QA_Sam"
'       • Sheet "Keys" with mapping table named "col_key" (preferred) or "col_keys"
'           Required headers: "QA_Sam_col", "Src_tbl_col"
'
' NOTES
'   - This macro appends to QA_Sam then removes fully blank rows, so re-runs don’t leave
'     leading empty rows.
'   - At the very start it forcibly clears any filters on QA_Sam so that previous runs
'     of the email macro (which filter QA_Sam) do not interfere with sampling.
'
' HOW TO RUN
'   - Put this code in a standard module of the MAIN workbook.
'   - Ensure all tables and headers exist and are named exactly as above.
'   - Run: Build_QA_Sample_Set
'=========================================================================================

Sub Build_QA_Sample_Set()

    '----------------------------
    ' Declarations
    '----------------------------
    Dim wb As Workbook
    Dim wsQA As Worksheet            ' QA Sample Set (QA_Sam lives here)
    Dim wsSrc As Worksheet           ' Source file (Src_tbl lives here)
    Dim wsKeys As Worksheet          ' Keys (mapping table lives here)
    Dim wsParams As Worksheet        ' QA_Parameters (tickler_count + F3 live here)

    Dim tblSrc As ListObject         ' Src_tbl
    Dim tblCount As ListObject       ' tickler_count
    Dim tblQA As ListObject          ' QA_Sam
    Dim tblMap As ListObject         ' col_key / col_keys

    Dim bodyRangeCount As Range
    Dim mapBody As Range

    Dim sampleSize As Long
    Dim colPct As Long, colSample As Long, colTicklerType As Long, srcTicklerCol As Long
    Dim mapQAColIdx As Long, mapSrcColIdx As Long

    Dim r As Long, i As Long, k As Long
    Dim tt As String, pct As Double, needed As Long
    Dim mapCount As Long
    Dim QAColIndex() As Long, SrcColIndex() As Long
    Dim qaHeader As String, srcHeader As String

    Dim matchRows() As Long, matchCount As Long
    Dim swapIdx As Long, tmp As Long
    Dim newRow As ListRow
    Dim totalSampled As Long

    On Error GoTo ErrHandler

    Set wb = ThisWorkbook
    Set wsQA = wb.Worksheets("QA Sample Set")
    Set wsSrc = wb.Worksheets("Source file")
    Set wsKeys = wb.Worksheets("Keys")
    Set wsParams = wb.Worksheets("QA_Parameters")

    Set tblSrc = wsSrc.ListObjects("Src_tbl")
    Set tblCount = wsParams.ListObjects("tickler_count")
    Set tblQA = wsQA.ListObjects("QA_Sam")

    ' Mapping table: prefer "col_key", fallback to "col_keys"
    On Error Resume Next
    Set tblMap = wsKeys.ListObjects("col_key")
    If tblMap Is Nothing Then Set tblMap = wsKeys.ListObjects("col_keys")
    On Error GoTo ErrHandler
    If tblMap Is Nothing Then
        MsgBox "Mapping table 'col_key' (or 'col_keys') not found on sheet 'Keys'.", vbCritical
        GoTo CleanExit
    End If

    '--------------------------------------------------------
    ' FIRST: ensure QA_Sam is completely UNFILTERED
    ' (macro 4/email macro may have left filters on QA_Sam)
    '--------------------------------------------------------
    SafeShowAllData_Table tblQA

    ' Performance
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual

    '--------------------------------------------------------
    ' 1) Clear QA_Sam contents (keep structure/validations)
    '--------------------------------------------------------
    If Not tblQA.DataBodyRange Is Nothing Then
        tblQA.DataBodyRange.ClearContents
    End If

    totalSampled = 0

    '--------------------------------------------------------
    ' 2) Compute Sample Set Count in QA_Parameters!tickler_count
    '    Sample size cell is QA_Parameters!F3 (must be numeric)
    '--------------------------------------------------------
    If IsNumeric(wsParams.Range("F3").Value) Then
        sampleSize = CLng(wsParams.Range("F3").Value)
    Else
        sampleSize = 0
    End If
    If sampleSize <= 0 Then
        MsgBox "Invalid or missing Sample Set Size in 'QA_Parameters'!F3.", vbCritical
        GoTo CleanExit
    End If

    colPct = GetTableColumnIndex(tblCount, "% of Total")
    colSample = GetTableColumnIndex(tblCount, "Sample Set Count")
    colTicklerType = GetTableColumnIndex(tblCount, "Tickler Type")

    If colPct = 0 Or colSample = 0 Or colTicklerType = 0 Then
        MsgBox "Required headers missing in 'tickler_count' (need 'Tickler Type', '% of Total', 'Sample Set Count').", vbCritical
        GoTo CleanExit
    End If

    If tblCount.DataBodyRange Is Nothing Then
        MsgBox "No data rows found in 'tickler_count' on 'QA_Parameters'.", vbCritical
        GoTo CleanExit
    End If
    Set bodyRangeCount = tblCount.DataBodyRange

    ' Calculate Sample Set Count = ROUNDUP(% of Total * F3)
    For r = 1 To bodyRangeCount.Rows.Count
        pct = 0
        If IsNumeric(bodyRangeCount.Cells(r, colPct).Value) Then
            pct = CDbl(bodyRangeCount.Cells(r, colPct).Value)
        End If

        If pct > 0 Then
            bodyRangeCount.Cells(r, colSample).Value = _
                Application.WorksheetFunction.RoundUp(pct * sampleSize, 0)
        Else
            bodyRangeCount.Cells(r, colSample).Value = 0
        End If
    Next r

    '--------------------------------------------------------
    ' 3) Build column mapping: QA_Sam_col -> Src_tbl_col
    '--------------------------------------------------------
    mapQAColIdx = GetTableColumnIndex(tblMap, "QA_Sam_col")
    mapSrcColIdx = GetTableColumnIndex(tblMap, "Src_tbl_col")
    If mapQAColIdx = 0 Or mapSrcColIdx = 0 Then
        MsgBox "Mapping table must contain 'QA_Sam_col' and 'Src_tbl_col' headers.", vbCritical
        GoTo CleanExit
    End If
    If tblMap.DataBodyRange Is Nothing Then
        MsgBox "No rows found in mapping table on 'Keys'.", vbCritical
        GoTo CleanExit
    End If
    Set mapBody = tblMap.DataBodyRange

    mapCount = 0
    For r = 1 To mapBody.Rows.Count
        qaHeader = Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value))
        srcHeader = Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value))
        If qaHeader <> "" And srcHeader <> "" Then mapCount = mapCount + 1
    Next r
    If mapCount = 0 Then
        MsgBox "No valid QA_Sam_col → Src_tbl_col mappings found.", vbCritical
        GoTo CleanExit
    End If

    ReDim QAColIndex(1 To mapCount)
    ReDim SrcColIndex(1 To mapCount)

    k = 0
    For r = 1 To mapBody.Rows.Count
        qaHeader = Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value))
        srcHeader = Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value))
        If qaHeader <> "" And srcHeader <> "" Then
            k = k + 1

            ' Get QA_Sam column index
            QAColIndex(k) = GetTableColumnIndex(tblQA, qaHeader)
            If QAColIndex(k) = 0 Then
                MsgBox "QA_Sam column '" & qaHeader & "' not found.", vbCritical
                GoTo CleanExit
            End If

            ' Get Src_tbl column index
            SrcColIndex(k) = GetTableColumnIndex(tblSrc, srcHeader)
            If SrcColIndex(k) = 0 Then
                MsgBox "Src_tbl column '" & srcHeader & "' not found.", vbCritical
                GoTo CleanExit
            End If
        End If
    Next r

    '--------------------------------------------------------
    ' 4) Random sampling per Tickler Type → append to QA_Sam
    '--------------------------------------------------------
    srcTicklerCol = GetTableColumnIndex(tblSrc, "Tickler Type")
    If srcTicklerCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If

    Randomize

    For r = 1 To bodyRangeCount.Rows.Count

        tt = Trim$(CStr(bodyRangeCount.Cells(r, colTicklerType).Value))
        needed = 0
        If IsNumeric(bodyRangeCount.Cells(r, colSample).Value) Then
            needed = CLng(bodyRangeCount.Cells(r, colSample).Value)
        End If

        If tt <> "" And needed > 0 Then
            ' Collect matches from Src_tbl for this Tickler Type
            matchCount = 0
            Erase matchRows

            With tblSrc.DataBodyRange
                For i = 1 To .Rows.Count
                    If Trim$(CStr(.Cells(i, srcTicklerCol).Value)) = tt Then
                        matchCount = matchCount + 1
                        ReDim Preserve matchRows(1 To matchCount)
                        matchRows(matchCount) = i
                    End If
                Next i
            End With

            If matchCount > 0 Then
                If needed > matchCount Then needed = matchCount

                ' Partial Fisher-Yates shuffle to pick unique random rows
                For i = 1 To needed
                    Dim pick As Long
                    pick = i + Int((matchCount - i + 1) * Rnd)
                    tmp = matchRows(i): matchRows(i) = matchRows(pick): matchRows(pick) = tmp
                Next i

                ' Append selections
                For i = 1 To needed
                    Set newRow = tblQA.ListRows.Add
                    For k = 1 To mapCount
                        newRow.Range.Cells(1, QAColIndex(k)).Value = _
                            tblSrc.DataBodyRange.Cells(matchRows(i), SrcColIndex(k)).Value
                    Next k
                    totalSampled = totalSampled + 1
                Next i
            End If
        End If
    Next r

    '--------------------------------------------------------
    ' 5) Delete fully blank rows from QA_Sam (cleanup)
    '--------------------------------------------------------
    Dim lr As Long, c As Range, rowBlank As Boolean
    For lr = tblQA.ListRows.Count To 1 Step -1
        rowBlank = True
        For Each c In tblQA.ListRows(lr).Range.Cells
            If Len(c.Value) > 0 Then
                rowBlank = False
                Exit For
            End If
        Next c
        If rowBlank Then tblQA.ListRows(lr).Delete
    Next lr

    MsgBox "QA Sample Set built successfully." & vbCrLf & _
           "Total sampled rows: " & totalSampled, vbInformation

CleanExit:
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Application.Calculation = xlCalculationAutomatic
    Exit Sub

ErrHandler:
    MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical, "Build_QA_Sample_Set"
    Resume CleanExit

End Sub

'============================================================
' Helper: Get column index in a ListObject by header name
'   Returns 0 if not found.
'============================================================
Private Function GetTableColumnIndex(ByVal tbl As ListObject, ByVal headerName As String) As Long
    Dim i As Long, target As String
    target = LCase$(Trim$(headerName))
    For i = 1 To tbl.ListColumns.Count
        If LCase$(Trim$(tbl.ListColumns(i).Name)) = target Then
            GetTableColumnIndex = i
            Exit Function
        End If
    Next i
    GetTableColumnIndex = 0
End Function

'============================================================
' Helper: Safely clear any AutoFilter on a ListObject
'   Ensures the table is fully unfiltered without throwing
'   errors if no filter is currently applied.
'============================================================
Private Sub SafeShowAllData_Table(ByVal lo As ListObject)
    On Error Resume Next
    ' If AutoFilter is Nothing, toggle one on and then clear it
    If lo.AutoFilter Is Nothing Then
        lo.Range.AutoFilter
        lo.AutoFilter.ShowAllData
    Else
        lo.AutoFilter.ShowAllData
    End If
    On Error GoTo 0
End Sub