import sys
import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message=f"Error occurred in script [{0}] line number [{1}] error message [{2}]".format(
     Line_number=exc_tb.tb_lineno, file_name=str(file_name), error_message=str(error)
    )
    return error_message
 
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return f"Error occurred in script [{self.file_name}] line number [{self.line_number}] error message [{self.error_message}]"

  
