Option Explicit

Sub Build_QA_Sample_Set()

    '========================================================
    ' Macro 2: Build QA sample set using randomization
    '
    ' 1) Clear QA_Sam table data (only contents, keep structure/validations).
    ' 2) In tickler_count:
    '       Sample Set Count = ROUNDUP([% of Total] * D2, 0)
    ' 3) Use Keys!col_key / col_keys:
    '       QA_Sam_col -> Src_tbl_col mappings.
    ' 4) For each Tickler Type with Sample Set Count > 0:
    '       Randomly pick that many rows from Src_tbl and
    '       append to QA_Sam (ListRows.Add).
    ' 5) After filling, delete any completely blank rows
    '       from QA_Sam so data starts at the top.
    '========================================================

    Dim wb As Workbook
    Dim wsQA As Worksheet
    Dim wsSrc As Worksheet
    Dim wsKeys As Worksheet
    
    Dim tblSrc As ListObject
    Dim tblCount As ListObject
    Dim tblQA As ListObject
    Dim tblMap As ListObject
    
    Dim bodyRangeCount As Range
    Dim mapBody As Range
    
    Dim sampleSize As Long
    Dim colPct As Long
    Dim colSample As Long
    Dim colTicklerType As Long
    Dim srcTicklerCol As Long
    
    Dim mapQAColIdx As Long
    Dim mapSrcColIdx As Long
    
    Dim i As Long, r As Long, k As Long
    Dim tt As String
    Dim pct As Double
    Dim needed As Long
    
    Dim mapCount As Long
    Dim QAColIndex() As Long
    Dim SrcColIndex() As Long
    
    Dim matchRows() As Long
    Dim matchCount As Long
    Dim swapIdx As Long, tmp As Long
    
    Dim newRow As ListRow
    Dim totalSampled As Long
    
    Dim qaHeader As String
    Dim srcHeader As String

    On Error GoTo ErrHandler
    
    Set wb = ThisWorkbook
    Set wsQA = wb.Worksheets("QA Sample Set")
    Set wsSrc = wb.Worksheets("Source file")
    Set wsKeys = wb.Worksheets("Keys")
    
    Set tblSrc = wsSrc.ListObjects("Src_tbl")
    Set tblCount = wsQA.ListObjects("tickler_count")
    Set tblQA = wsQA.ListObjects("QA_Sam")
    
    ' Mapping table: try col_key then col_keys
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
    ' 1) Clear QA_Sam table data (keep structure & validations)
    '--------------------------------------------------------
    If Not tblQA.DataBodyRange Is Nothing Then
        tblQA.DataBodyRange.ClearContents
    End If
    
    totalSampled = 0

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
    
    colPct = GetTableColumnIndex(tblCount, "% of Total")
    colSample = GetTableColumnIndex(tblCount, "Sample Set Count")
    colTicklerType = GetTableColumnIndex(tblCount, "Tickler Type")
    
    If colPct = 0 Or colSample = 0 Or colTicklerType = 0 Then
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
    ' 3) Build column mapping: QA_Sam_col -> Src_tbl_col
    '--------------------------------------------------------
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
        qaHeader = Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value))
        srcHeader = Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value))
        If qaHeader <> "" And srcHeader <> "" Then
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
        qaHeader = Trim$(CStr(mapBody.Cells(r, mapQAColIdx).Value))
        srcHeader = Trim$(CStr(mapBody.Cells(r, mapSrcColIdx).Value))
        
        If qaHeader <> "" And srcHeader <> "" Then
            k = k + 1
            
            QAColIndex(k) = GetTableColumnIndex(tblQA, qaHeader)
            If QAColIndex(k) = 0 Then
                MsgBox "QA_Sam column '" & qaHeader & "' not found in QA_Sam table.", vbCritical
                GoTo CleanExit
            End If
            
            SrcColIndex(k) = GetTableColumnIndex(tblSrc, srcHeader)
            If SrcColIndex(k) = 0 Then
                MsgBox "Src_tbl column '" & srcHeader & "' not found in Src_tbl table.", vbCritical
                GoTo CleanExit
            End If
        End If
    Next r

    '--------------------------------------------------------
    ' 4) Random sampling per Tickler Type (append rows)
    '--------------------------------------------------------
    srcTicklerCol = GetTableColumnIndex(tblSrc, "Tickler Type")
    If srcTicklerCol = 0 Then
        MsgBox "'Tickler Type' column not found in Src_tbl.", vbCritical
        GoTo CleanExit
    End If
    
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.Calculation = xlCalculationManual
    
    Randomize
    
    ' Loop tickler_count rows
    For r = 1 To bodyRangeCount.Rows.Count
        
        tt = Trim$(CStr(bodyRangeCount.Cells(r, colTicklerType).Value))
        needed = 0
        If IsNumeric(bodyRangeCount.Cells(r, colSample).Value) Then
            needed = CLng(bodyRangeCount.Cells(r, colSample).Value)
        End If
        
        If tt <> "" And needed > 0 Then
            
            ' Collect matching Src_tbl rows for this Tickler Type
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
                
                If needed > matchCount Then
                    needed = matchCount
                End If
                
                ' Partial shuffle for random unique selection
                For i = 1 To needed
                    swapIdx = i + Int((matchCount - i + 1) * Rnd)
                    tmp = matchRows(i)
                    matchRows(i) = matchRows(swapIdx)
                    matchRows(swapIdx) = tmp
                Next i
                
                ' Append selected rows into QA_Sam
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
    ' 5) Delete completely blank rows from QA_Sam
    '    so table starts clean at first data row.
    '--------------------------------------------------------
    Dim lr As Long, c As Range, isBlank As Boolean
    
    For lr = tblQA.ListRows.Count To 1 Step -1
        isBlank = True
        For Each c In tblQA.ListRows(lr).Range.Cells
            If Len(c.Value) > 0 Then
                isBlank = False
                Exit For
            End If
        Next c
        
        If isBlank Then
            tblQA.ListRows(lr).Delete
        End If
    Next lr

    MsgBox "QA Sample Set built successfully." & vbCrLf & _
           "Total sampled rows: " & totalSampled, vbInformation

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
