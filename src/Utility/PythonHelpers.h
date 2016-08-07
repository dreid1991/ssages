#include "Python.h"
#include <boost/python.hpp>
namespace py = boost::python;
namespace SSAGES {
namespace PythonHelpers {
void printErrors() {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    py::handle<> hType(ptype);
    py::object extype(hType);

    py::handle<> hTraceback(ptraceback);
    py::object traceback(hTraceback);

    std::string errorMsg = py::extract<std::string>(pvalue);
    int lineNum = py::extract<int>(traceback.attr("tb_lineno"));
    std::string funcname = py::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_name"));
    std::string filename = py::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
    std::cout << "Error in python script" << std::endl;
    std::cout << errorMsg << std::endl;
    std::cout << "on line " << lineNum << std::endl;
    std::cout << "in function " << funcname << std::endl;
    std::cout << "in file " << filename << std::endl;
    //char *err = PyString_AsString(pvalue);
    //std::cout << err << std::endl;
    exit(0);
}
}
}
