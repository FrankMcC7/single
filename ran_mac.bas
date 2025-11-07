Sub GenerateRandomValue()
    Dim targetCell As Range
    Set targetCell = ThisWorkbook.Sheets("Sheet1").Range("B5") ' Hard-coded cell
    
    ' Generate random integer between 20 and 50
    Randomize
    targetCell.Value = Int((50 - 20 + 1) * Rnd + 20)
End Sub

Sub GenerateRandomValue()
    Dim targetCell As Range
    Set targetCell = ThisWorkbook.Sheets("Sheet1").Range("B5") ' hard-coded cell

    Randomize
    targetCell.Value = Int((50 - 20 + 1) * Rnd + 20)
End Sub

Sub ButtonRandom_Click()
    ' Shape click handler
    Call GenerateRandomValue
End Sub

Sub CreateModernRoundedButton()
    Dim shp As Shape
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("Sheet1") ' change if needed

    ' Delete old button with same name (optional safety)
    On Error Resume Next
    ws.Shapes("btnRandom").Delete
    On Error GoTo 0

    ' Create rounded rectangle
    Set shp = ws.Shapes.AddShape(msoShapeRoundedRectangle, 100, 50, 170, 34)
    
    With shp
        .Name = "btnRandom"
        .TextFrame2.TextRange.Text = "Generate Random Value"
        
        ' Rounded corners
        .Adjustments.Item(1) = 0.3   ' 0 = square, 1 = max curve

        ' Fill & border (modern flat look)
        .Fill.ForeColor.RGB = RGB(52, 152, 219)         ' blue
        .Line.ForeColor.RGB = RGB(41, 128, 185)         ' darker border
        .Line.Weight = 1.2

        ' Text styling
        With .TextFrame2.TextRange.Font
            .Name = "Segoe UI"
            .Size = 10
            .Bold = msoTrue
            .Fill.ForeColor.RGB = RGB(255, 255, 255)    ' white
        End With

        ' Center text nicely
        .TextFrame2.VerticalAnchor = msoAnchorMiddle
        .TextFrame2.TextRange.ParagraphFormat.Alignment = msoAlignCenter

        ' Assign macro on click
        .OnAction = "ButtonRandom_Click"
    End With
End Sub