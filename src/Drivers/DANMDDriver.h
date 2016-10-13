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
#include "../hooks/gpu_md/FixSSAGES.h"
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
            py::object main = py::import("__main__");
            py::object globals = main.attr("__dict__");
            PyObject *runSimulation = PyDict_GetItemString(globals.ptr(), runFuncName.c_str());
            if (runSimulation == (PyObject *) NULL) {
                std::cout << "No runsimulation function in python script.  Must be global variable called " << runFuncName << std::endl;
                exit(0);
            }
            if (not PyCallable_Check(runSimulation)) {
                std::cout << "Global variable with name " << runFuncName << " is not callable " << std::endl;
                exit(0);

            }
            try {
                py::call<void>(runSimulation, _statePy, _MDsteps);
            } catch (py::error_already_set &) {
                PythonHelpers::printErrors();
            }
		}

		virtual void ExecuteInputFile(std::string contents) override
		{
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
                printf("I'M GOING TO CALL SETUP\n");
                _statePy = py::call<py::object>(setupSimulation);
                std::string asStr = py::extract<std::string>(py::str(_statePy));
                std::cout << "Received the following from " << setupFuncName << ":" << std::endl;
                std::cout << asStr << std::endl;
                _state = py::extract<boost::shared_ptr<State> >(_statePy);
            } catch (py::error_already_set &) {
                PythonHelpers::printErrors();
            }
            std::string handle = "ssages fix";
            boost::shared_ptr<FixSSAGES> fixSSAGES = boost::shared_ptr<FixSSAGES>(new FixSSAGES(_state, handle));
            _state->activateFix(fixSSAGES);
            if(!(_hook = dynamic_cast<Hook*>(fixSSAGES.get())))
            {
                throw BuildException({"Unable to dynamic cast hook on node " + std::to_string(_world.rank())});			
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
