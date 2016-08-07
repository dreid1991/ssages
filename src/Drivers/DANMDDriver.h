#pragma once
#include "Python.h"
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "Drivers/Driver.h"
#include "../Validator/ObjectRequirement.h"
#include "../include/schema.h"
#include "../Utility/BuildException.h"
#include "../Utility/PythonHelpers.h"
#include "State.h"
namespace mpi = boost::mpi;
using namespace Json;
namespace py = boost::python;
namespace SSAGES
{
	class DANMDDriver : public Driver 
	{
	private:

		//pointer to this local instance of lammps

		// The number of MD engine steps you would like to perform
		int _MDsteps;

		// The lammps logfile
		std::string _logfile;
        std::string setupFuncName;
        std::string runFuncName;
        boost::shared_ptr<State> _state;
        py::object _statePy;

	public:

		DANMDDriver(mpi::communicator& world_comm,
					 mpi::communicator& local_comm,
					 int walkerID) : 
		Driver(world_comm, local_comm, walkerID), _MDsteps(), _logfile(), setupFuncName("setupSimulation"), runFuncName("runSimulation")
		{
		};

		virtual void Run() override
		{
			std::string rline = "run " + std::to_string(_MDsteps);
			//_lammps->input->one(rline.c_str());
		}

		// Run LAMMPS input file line by line and gather the fix/hook
		virtual void ExecuteInputFile(std::string contents) override
		{
		    //here we'll call like boost::python::exec or eval with the contents as the arg and the state we constructed as an environment variable
            py::object main = py::import("__main__");
            py::object globals = main.attr("__dict__");
            try {
                py::exec(contents.c_str(), globals);
            } catch (py::error_already_set &) {
                PythonHelpers::printErrors();
            }
            //okay, now global funcs setupSimulation and runSimulation should be defined.  Check for this.
            //char *err = PyString_AsString(pvalue);
            PyObject *setupSimulation = PyDict_GetItemString(globals.ptr(), setupFuncName.c_str());
            if (setupSimulation == (PyObject *) NULL) {
                std::cout << "No setup simulation function in python script.  Must be global variable called " << setupFuncName << std::endl;
                exit(0);
            }
            if (not PyCallable_Check(setupSimulation)) {
                std::cout << "Global variable with name " << setupFuncName << " is not callable " << std::endl;
                exit(0);

            }
            try {
                py::object _statePy = py::call<py::object>(setupSimulation);
                std::string asStr = py::extract<std::string>(py::str(_statePy));
                std::cout << "Received the following from " << setupFuncName << ":" << std::endl;
                std::cout << asStr << std::endl;
                _state = py::extract<boost::shared_ptr<State> >(_statePy);
            } catch (py::error_already_set &) {
                PythonHelpers::printErrors();
            }

		}

		virtual void BuildDriver(const Json::Value& json, const std::string& path) override
		{

			Value schema;
			ObjectRequirement validator;
			Reader reader;

			reader.parse(JsonSchema::DANMDDriver, schema);
			validator.Parse(schema, path);

			// Validate inputs.
			validator.Validate(json, path);
			if(validator.HasErrors())
				throw BuildException(validator.GetErrors());

			_MDsteps = json.get("MDSteps",1).asInt();
			_inputfile = json.get("inputfile","none").asString();
            
            //initialize!
            Py_Initialize();
            //py::object main = py::import("__main__");
            //py::object global(main.attr("__dict__"));
            
//            _state = boost::shared_ptr<State>(new State());

            
		    	

		}

		// Serialize
		virtual void Serialize(Json::Value& json) const override
		{
			json["MDSteps"] = _MDsteps;
			json["logfile"] = _logfile;
			json["type"] = "LAMMPS";
			json["number processors"] = _comm.size();
			if(_inputfile != "none")
				json["inputfile"] = _inputfile;

			// Need CVs and Methods still
			

		}
	};
}
