#pragma once
#include "Fix.h"
#include "Hook.h"
#include <string>
#include "GPUArrayGlobal.h"
namespace SSAGES {
	// SSAGES Hook class for LAMMPS implemented as 
	// a LAMMPS fix. This is activated by adding 
	// a "ssages" fix to "all". Note that thermo must 
	// be set to 1 in order for the synchronizing to work.
	class FixSSAGES : public Fix, public SSAGES::Hook
	{
	protected:
		void SyncToEngine();

		void SyncToSnapshot();


	public:
		FixSSAGES(SHARED(State) state, std::string handle);

		// Setup for presimulation call.
        
  	    bool prepareForRun();
		// Post force where the synchronization occurs.
  		void compute(bool);
        bool postRun();
        GPUArrayGlobal<uint> _activeIds;
        GPUArrayGlobal<char> _dataBuffer;
        GPUArrayGlobal<float4> _forceBuffer;

  		// Post-run for post-simulation call.
	};
};
