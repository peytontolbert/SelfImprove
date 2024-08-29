import openpyxl
import json

class SpreadsheetManager:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            self.workbook = openpyxl.load_workbook(file_path)
        except FileNotFoundError:
            self.workbook = openpyxl.Workbook()
            self.workbook.save(file_path)
        self.sheet = self.workbook.active

    def read_data(self, cell_range):
        """Read data from a specified cell range."""
        data = []
        for row in self.sheet[cell_range]:
            data.append([cell.value for cell in row])
        return data

    def write_data(self, start_cell, data):
        """Write data to the spreadsheet starting from a specified cell."""
        for row_idx, row_data in enumerate(data, start=start_cell[0]):
            for col_idx, value in enumerate(row_data, start=start_cell[1]):
                # Convert complex data types to JSON strings
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                self.sheet.cell(row=row_idx, column=col_idx, value=str(value))
        self.workbook.save(self.file_path)

    def add_sheet(self, sheet_name):
        """Add a new sheet to the workbook."""
        self.workbook.create_sheet(title=sheet_name)
        self.workbook.save(self.file_path)
