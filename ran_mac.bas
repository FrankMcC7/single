Sub GenerateRandomValue()
    Dim targetCell As Range
    Set targetCell = ThisWorkbook.Sheets("Sheet1").Range("B5") ' Hard-coded cell
    
    ' Generate random integer between 20 and 50
    Randomize
    targetCell.Value = Int((50 - 20 + 1) * Rnd + 20)
End Sub