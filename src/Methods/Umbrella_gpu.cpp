#include "Umbrella_gpu.h"
#include "Umbrella_gpu_kernels.h"
#include <iostream>

namespace SSAGES
{

	void Umbrella_gpu::PreSimulation(Snapshot*, const CVList& cvs)
	{
		if(_comm.rank() == 0)
		{
		 	_umbrella.open(_filename.c_str(), std::ofstream::out | std::ofstream::app);
		 }
	}

	void Umbrella_gpu::PostIntegration(Snapshot* snapshot, const CVList& cvs)
	{
		// Compute the forces on the atoms from the CV's using the chain 
		// rule.
        
		for(size_t i = 0; i < cvs.size(); ++i)
		{
			// Get current CV and gradient.
			auto& cv = cvs[i];
            float *val = cv->GetValue_gpu();
			float4 *grad = cv->GetGradient_gpu();
            float4 *fs = snapshot->_gpd.fs;
            float center = _centers[i];
            float k = _kspring[i];
            int nAtoms = snapshot->_gpd.nAtoms;
            
            call_umbrella_eval(fs, val, grad, center, k, nAtoms);
		}

		_iteration++;
		if(_iteration % _logevery == 0)
			PrintUmbrella(cvs);
	}

	void Umbrella_gpu::PostSimulation(Snapshot*, const CVList&)
	{
		if(_comm.rank() ==0)
		{
			_umbrella.close();
		}
	}

	void Umbrella_gpu::PrintUmbrella(const CVList& CV)
	{
//          TODO
		if(_comm.rank() ==0)
		{
			_umbrella.precision(8);
			_umbrella << _iteration << " ";

			for(size_t jj = 0; jj < _centers.size(); jj++)
				_umbrella/*<< _centers[jj] << " " */<< CV[jj]->GetValue()<< " "; 

			_umbrella<<std::endl;
		}
        
	}
}
