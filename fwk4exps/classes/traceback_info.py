import traceback
import sys


class TraceBackInfo(object):
    def getExperimentState():
        """
        Funcion para obtener el 'Estado' de una funcion
        Se debe tener en cuenta el numero de llamadas a funcion
        previos a esta funcion en este caso -3 indica que hace 3 funciones
        se llamo a  la funcion a la cual le queremos sacar su estado.
        el estado se compone de sus variables locales y el numero de linea
        donde fue llamado, estos valores son transformados
        en un string que se retorna para formar la llave del
        nodo.
        """
        #print("getexperiment info")
        #  ----------Traceback info:
        extracted_list = traceback.extract_stack()
        formated_traceback_list = traceback.format_list(extracted_list)
        #  ----------Formated traceback list
        important_line = formated_traceback_list[-3]
        #print("important_line:")
        #print(important_line)
        #print("line_no:")
        line_no = extracted_list[-3][1]
        #print(line_no)

        #print("local variables from experimentalDesign:")
        call_frame = sys._getframe(2)
        eval_locals = call_frame.f_locals
        #print(eval_locals)

        return_str = str(line_no)+str(eval_locals)

        return return_str
