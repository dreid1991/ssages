#pragma once
#include "Fix.h"
#include "Hook.h"
#include <string>
namespace SSAGES {
	// SSAGES Hook class for LAMMPS implemented as 
	// a LAMMPS fix. This is activated by adding 
	// a "ssages" fix to "all". Note that thermo must 
	// be set to 1 in order for the synchronizing to work.
	class FixSSAGES : public Fix, SSAGES::Hook
	{
	protected:
		void SyncToEngine();

		void SyncToSnapshot();

		void SyncToEngineCPU();

		void SyncToSnapshotCPU();


	public:
		FixSSAGES(SHARED(State) state, std::string handle);

		// Setup for presimulation call.
        
  	    bool prepareForRun();
		// Post force where the synchronization occurs.
  		void compute(bool);
        bool postRun();

  		// Post-run for post-simulation call.
	};
};
