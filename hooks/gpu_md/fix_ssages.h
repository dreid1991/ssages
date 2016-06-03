#pragma once
#include "Fix.h"
#include "Hook.h"

	// SSAGES Hook class for LAMMPS implemented as 
	// a LAMMPS fix. This is activated by adding 
	// a "ssages" fix to "all". Note that thermo must 
	// be set to 1 in order for the synchronizing to work.
	class FixSSAGES : public Fix, SSAGES::Hook
	{
	protected:
		// Implementation of the SyncToEngine interface.
		void SyncToEngine() override;

		// Implementation of the SyncToSnapshot interface.
		void SyncToSnapshot() override;

	public:
		FixSSAGES(SHARED(State) state, string handle);

		// Setup for presimulation call.
        
  	    bool prepareForRun();
		// Post force where the synchronization occurs.
  		void compute(bool);
        void postRun();

  		// Post-run for post-simulation call.
	};

