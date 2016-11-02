#pragma once 

#include "Method.h"
#include "../CVs/CollectiveVariable.h"
#include <fstream>
#include "GPUArrayGlobal.h"

namespace SSAGES
{
	// Umbrella sampling method to constrain an arbitrary 
	// number of CVs at specified equilibrium distances.
	class Umbrella_gpu : public Method
	{
	private:
		// Vector of spring constants.
        std::vector<double> _kspring;


		// Vector of equilibrium distances.
        std::vector<double> _centers;


		//! File name
		std::string _filename;

		//! Log every n time steps
		int _logevery;
        
        // Output stream for umbrella data.
		std::ofstream _umbrella;

	public:
		// Create instance of umbrella with spring constants "kspring", 
		// and centers "centers". Note the sizes of the vectors should be 
		// commensurate with the number of CVs.
		Umbrella_gpu(boost::mpi::communicator& world,
				 boost::mpi::communicator& comm,
				 const std::vector<double>& ksprings,
				 const std::vector<double>& centers,
				 std::string name,               
				 unsigned int frequency) : 
		Method(frequency, world, comm), _kspring(ksprings), _centers(centers),
		_filename(name), _logevery(1)
		{}

		// Pre-simulation hook.
		void PreSimulation(Snapshot* snapshot, const CVList& cvs) override;

		// Post-integration hook.
		void PostIntegration(Snapshot* snapshot, const CVList& cvs) override;

		// Post-simulation hook.
		void PostSimulation(Snapshot* snapshot, const CVList& cvs) override;
		
		void PrintUmbrella(const CVList& cvs);

        void SetLogStep(const int iter)
		{
			_logevery = iter;
		}
        
		void Serialize(Json::Value& json) const override
		{
			json["type"] = "Umbrella";
			for(auto& k : _kspring)
				json["ksprings"].append(k);

			for(auto& c : _centers)
				json["centers"].append(c);

			json["file name"] = _filename;

			json["iteration"] = _iteration;

			json["log every"] = _logevery;
		}

	};
}
