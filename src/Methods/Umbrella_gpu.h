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
        std::vector<double> _ksprings;


		// Vector of equilibrium distances.
        std::vector<double> _centers;

		// iterator for this method
		int _currentiter;

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
				 unsigned int frequency) : 
		Method(frequency, world, comm), _ksprings(ksprings), _centers(centers)
		{}

		// Pre-simulation hook.
		void PreSimulation(Snapshot* snapshot, const CVList& cvs) override;

		// Post-integration hook.
		void PostIntegration(Snapshot* snapshot, const CVList& cvs) override;

		// Post-simulation hook.
		void PostSimulation(Snapshot* snapshot, const CVList& cvs) override;
		
		void PrintUmbrella(const CVList& cvs);

		void Serialize(Json::Value& json) const override
		{

		}

	};
}
