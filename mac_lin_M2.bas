Option Explicit

Sub Build_QA_Sample_Set()

    '========================================================
    ' Macro 2: Build QA sample set using randomization
    '
    ' 1) Clear data rows (only contents) of QA_Sam table.
    ' 2) In tickler_count, set [Sample Set Count] =
    '       ROUNDUP([% of Total] * Sample_Size_D2, 0)
    ' 3) Use Keys!col_key (QA_Sam_col -> Src_tbl_col) mappings
    '    to know which Src_tbl columns populate which QA_Sam cols.
    ' 4) For each Tickler Type:
    '       Randomly select N rows from Src_tbl where
    '       N = Sample Set Count for that Tickler Type
    '       (no replacement within that tickler type),
    '       and write mapped fields into QA_Sam.
    '========================================================

    Dim wb As Workbook
    Dim wsQA As Worksheet
    Dim wsSrc As Worksheet
    Dim wsKeys As Worksheet
    
    Dim tblSrc As ListObject
    Dim tblCount As ListObject
    Dim tblQA As ListObject
    Dim tblMap As ListObject
    
    Dim bodyRangeQA As Range
    Dim bodyRangeCount As Range
    Dim mapBody As Range
    
    Dim sampleSize As Long
    Dim colPct As Long
    Dim colSample As Long
    Dim colTicklerCount As Long
    Dim srcTicklerCol As Long
    
    Dim mapQAColIdx As Long
    Dim mapSrcColIdx As Long
    
    Dim i As Long, r As Long, k As Long
    Dim lastRow As Long
    Dim tt As String
    Dim pct As Double
    Dim needed As Long
    
    Dim mapCount As Long
    Dim QAColIndex() As Long
    Dim SrcColIndex() As Long
    
    Dim matchRows() As Long
    Dim matchCount As Long
    Dim idx As Long, swapIdx As Long, tmp As Long
    
    Dim newRow As ListRow
    Dim dictSamples As Object ' not strictly required but reserved
    Dim uniqueNeeded As Long
    Dim maxQARows As Long
    Dim currentQARows As Long
    
    On Error GoTo ErrHandler
    
    Set wb = ThisWorkbook
    Set wsQA = wb.Worksheets("QA Sample Set")
    Set wsSrc = wb.Worksheets("Source file")
    Set wsKeys = wb.Worksheets("Keys")
    
    Set tblSrc = wsSrc.ListObjects("Src_tbl")
    Set tblCount = wsQA.ListObjects("tickler_count")
    Set tblQA = wsQA.ListObjects("QA_Sam")
    
    ' Try col_key first, fallback to col_keys if needed
    On Error Resume Next
    Set tblMap = wsKeys.ListObjects("col_key")
    If tblMap Is Nothing Then
        Set tblMap = wsKeys.ListObjects("col_keys")
    End If
    On Error GoTo ErrHandler
    
    If tblMap Is Nothing Then
        MsgBox "Mapping table 'col_key' (or 'col_keys') not found on 'Keys' sheet.", vbCritical
        GoTo CleanExit
    End If
    
    '--------------------------------------------------------
    ' 1) Clear QA_Sam table body (keep validations & structure)
    '--------------------------------------------------------
    If Not tblQA.DataBodyRange Is Nothing Then
        tblQA.DataBodyRange.ClearContents
    End If
    
    ' We'll append fresh sample rows; ensure we know starting point
    currentQARows = 0
    If Not tblQA.DataBodyRange Is Nothing Then
        currentQARows = tblQA.DataBodyRange.Rows.Count
    End If
    
    '--------------------------------------------------------
    ' 2) Populate Sample Set Count in tickler_count
    '--------------------------------------------------------
    sampleSize = 0
    If IsNumeric(wsQA.Range("D2").Value) Then
        sampleSize = CLng(wsQA.Range("D2").Value)
    End If
    
    If sampleSize <= 0 Then
        MsgBox "Invalid or missing Sample Set Size in 'QA Sample Set'!D2.", vbCritical
        GoTo CleanExit
    End If
    
    ' Identify columns in tickler_count
    colPct = GetTableColumnIndex(tblCount, "% of Total")
    colSample = GetTableColumnIndex(tblCount, "Sample Set Count")
    colTicklerCount = GetTableColumnIndex(tblCount, "Tickler Type")
    
    If colPct = 0 Or colSample = 0 Or colTicklerCount = 0 Then
        MsgBox "Required columns ('Tickler Type', '% of Total', 'Sample Set Count') not found in 'tickler_count' table.", vbCritical
        GoTo CleanExit
    End If
    
    If tblCount.DataBodyRange Is Nothing Then
        MsgBox "No data rows in 'tickler_count' table.", vbCritical
        GoTo CleanExit
    End If
    
    Set bodyRangeCount = tblCount.DataBodyRange
    
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
    ' 3) Build column mapping from Keys!col_key
    '     QA_Sam_col -> Src_tbl_col
    '--------------------------------------------------------
    ' Find mapping columns
    mapQAColIdx = GetTableColumnIndex(tblMap, "QA_Sam_col")
    mapSrcColIdx = GetTableColumnIndex(tblMap, "Src_tbl_col")
    
    If mapQAColIdx = 0 Or mapSrcColIdx = 0 Then
        MsgBox "Columns 'QA_Sam_col' and/or 'Src_tbl_col' not found in mapping table.", vbCritical
        GoTo CleanExit
    End If
    
    If tblMap.DataBodyRange Is Nothing Then
        MsgBox "No rows in mapping table on 'Keys' sheet.", vbCritical
        GoTo CleanExit
    End If
    
    Set mapBody = tblMap.DataBodyRange
    
    ' Count valid mappings
    mapCount = 0
    For r = 1 To mapBody.Rows.Count
        If Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value)) <> "" And _
           Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value)) <> "" Then
            mapCount = mapCount + 1
        End If
    Next r
    
    If mapCount = 0 Then
        MsgBox "No valid QA_Sam_col -> Src_tbl_col mappings defined in Keys table.", vbCritical
        GoTo CleanExit
    End If
    
    ReDim QAColIndex(1 To mapCount)
    ReDim SrcColIndex(1 To mapCount)
    
    k = 0
    For r = 1 To mapBody.Rows.Count
        Dim qaHeader As String
        Dim srcHeader As String
        qaHeader = Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value))
        srcHeader = Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value))
        
        If qaHeader <> "" And srcHeader <> "" Then
            k = k + 1
            QAColIndex(k) = GetTableColumnIndex(tblQA, qaHeader)
            SrcColIndex(k) = GetTableColumnIndex(tblSrc, srcHeader)
            
            If QAColIndex(k) = 0 Then
                MsgBox "QA_Sam column '" & qaHeader & "' not found in QA_Sam table.", vbCritical
                GoTo CleanExit
            End If
            
            If SrcColIndex(k) = 0 Then
                MsgBox "Src_tbl column '" & srcHeader & "' not found in Src_tbl table.", vbCritical
                GoTo CleanExit
            End If
        End If
    Next r
    
    '--------------------------------------------------------
    ' 4) Random sampling per Tickler Type
    '--------------------------------------------------------
    srcTicklerCol = GetTableColumnIndex(tblSrc, "Tickler Type")
    If srcTicklerCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If
    
    Randomize ' Seed RNG
    
    maxQARows = 0 ' track total rows added, mainly informational
    
    ' Loop through each Tickler Type row in tickler_count
    For r = 1 To bodyRangeCount.Rows.Count
        
        tt = Trim$(CStr(bodyRangeCount.Cells(r, colTicklerCount).Value))
        needed = 0
        If IsNumeric(bodyRangeCount.Cells(r, colSample).Value) Then
            needed = CLng(bodyRangeCount.Cells(r, colSample).Value)
        End If
        
        If tt <> "" And needed > 0 Then
            
            ' Collect all Src_tbl rows matching this Tickler Type
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
            
            If matchCount = 0 Then
                ' No matching rows: skip this tickler type
            Else
                ' Cap needed to available rows
                If needed > matchCount Then
                    needed = matchCount
                End If
                
                ' Partial Fisher-Yates shuffle to get unique random rows
                For i = 1 To needed
                    swapIdx = i + Int((matchCount - i + 1) * Rnd)
                    tmp = matchRows(i)
                    matchRows(i) = matchRows(swapIdx)
                    matchRows(swapIdx) = tmp
                Next i
                
                ' Take first [needed] entries from shuffled list
                For i = 1 To needed
                    ' Add row to QA_Sam
                    Set newRow = tblQA.ListRows.Add
                    maxQARows = maxQARows + 1
                    
                    ' Populate mapped columns
                    For k = 1 To mapCount
                        newRow.Range.Cells(1, QAColIndex(k)).Value = _
                            tblSrc.DataBodyRange.Cells(matchRows(i), SrcColIndex(k)).Value
                    Next k
                Next i
            End If
        End If
    Next r
    
    MsgBox "QA Sample Set built successfully." & vbCrLf & _
           "Total sampled rows: " & maxQARows, vbInformation

CleanExit:
    Application.ScreenUpdating = True
    Application.EnableEvents = True
    Application.Calculation = xlCalculationAutomatic
    Application.DisplayAlerts = True
    Exit Sub

ErrHandler:
    MsgBox "Error " & Err.Number & ": " & Err.Description, vbCritical, "Build_QA_Sample_Set"
    Resume CleanExit

End Sub

'============================================================
' Helper: Get column index in a ListObject by header name
'============================================================
Private Function GetTableColumnIndex(ByVal tbl As ListObject, ByVal headerName As String) As Long
    Dim i As Long
    Dim target As String
    target = LCase$(Trim$(headerName))
    
    For i = 1 To tbl.ListColumns.Count
        If LCase$(Trim$(tbl.ListColumns(i).Name)) = target Then
            GetTableColumnIndex = i
            Exit Function
        End If
    Next i
    
    GetTableColumnIndex = 0
End Function
