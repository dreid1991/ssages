#include <iostream>

#include "fix_ssages.h"
#include "Methods/Meta.h"
#include "CVs/AtomCoordinateCV.h"
#include "State.h"
//HEY - YOU NEED TO ADD force_last AS ONE OF THE PER-ATOM VARIABLES IF YOU WANT TO BE ABLE TO FULLY TRANSFER STATES
using namespace SSAGES;

{
	FixSSAGES::FixSSAGES(SHARED(State) state, string handle) 
	Fix(state, handle, "all", "fillInATypePlease", false, false, false, 1),
    Hook()
	{
		///////Test Umbrella//////////////////////////////
		//this->AddListener(new Umbrella({100.0}, {0}, 1));
		//this->AddCV(new AtomCoordinateCV(1, 0));

		///////Test MetaDynamics//////////////////////////
	//	this->AddListener(new Meta(0.5, {0.05, 0.05}, 500, 1));
	//	this->AddCV(new AtomCoordinateCV(1, 0));
	//	this->AddCV(new AtomCoordinateCV(1, 1));
	}

	void FixSSAGES::prepareForRun()
	{
        //so this is always called while data is on cpu
		// Allocate vectors for snapshot.
		auto n = state->atoms.size();
		auto& pos = _snapshot.GetPositions();
		pos.resize(n);
		auto& vel = _snapshot.GetVelocities();
		vel.resize(n);
		auto& frc = _snapshot.GetForces();
		frc.resize(n);
		auto& ids = _snapshot.GetAtomIDs();
		ids.resize(n);
		auto& types = _snapshot.GetAtomTypes();
		types.resize(n);

		SyncToSnapshot();
		Hook::PreSimulationHook();
        //commented out because not sure where to store SnapshotGPUData yes
      //  _snapshot._gpd.xsFormatSame = true;
      //  _snapshot._gpd.vsFormatSame = true;
      //  _snapshot._gpd.fsFormatSame = true;
      //  _snapshot._gpd.idsFormatSame = true;
      //  _snapshot._gpd.typesFormatSame = true;
      //  _snapshot._gpd.nAtoms = n;
	}

	void FixSSAGES::compute()
	{
        syncToSnapshotRuntime();
		Hook::PostIntegrationHook();
	}

	void FixSSAGES::postRun()
	{
        //data is back on cpu at this point
		SyncToSnapshot();
		Hook::PostSimulationHook();
	}
    void FixSSAGES::syncToSnapshotRuntime() {
        GPUData &gpd = state->gpd;
        //commented out because not sure where to store SnapshotGPUData yes
       // _snapshot._gpd.xs = gpd.xs.getDevData()
       // _snapshot._gpd.vs = gpd.vs.getDevData()
       // _snapshot._gpd.fs = gpd.fs.getDevData()
       // _snapshot._gpd.ids = gpd.ids.getDevData()
       // _snapshot._gpd.types = gpd.types.getDevData()
    }
    void FixSSAGES::syncToEngineRuntime() {
        //nothing necessary b/c all data formats conform
    }
	void FixSSAGES::SyncToSnapshot()
	{

        //this is only called when data in cpu, so between runs (like when exchanging walkers or something)	
		const auto* _atom = atom;

		auto& pos = _snapshot.GetPositions();
		auto& vel = _snapshot.GetVelocities();
		auto& frc = _snapshot.GetForces();

		// Labels and ids for future work on only updating
		// atoms that have changed.
		auto& ids = _snapshot.GetAtomIDs();
		auto& types = _snapshot.GetAtomTypes();

		// Get iteration.
		_snapshot.SetIteration(state->turn);
		
		// Get volume.
		double vol = state->bounds.volume();
		_snapshot.SetVolume(vol);

		// Update values.
        for (Atom &a : state->atoms) {
            pos[i] = atom.pos;
            vel[i] = atoms.vel;
            frc[i] = atom.force;

            ids[i] = atom.id;
            types[i] = atom.type;
			
		}	}

	void FixSSAGES::SyncToEngine() //put Snapshot values -> LAMMPS
	{
		// Obtain local const reference to snapshot variables.
		// Const will ensure that _snapshot variables are
		// not being changed. Only engine side variables should
		// change. 
		const auto& pos = _snapshot.GetPositions();
		const auto& vel = _snapshot.GetVelocities();
		const auto& frc = _snapshot.GetForces();

		// Labels and ids for future work on only updating
		// atoms that have changed.
		const auto& ids = _snapshot.GetAtomIDs();
		const auto& types = _snapshot.GetAtomTypes();

		// Loop through all atoms and set their values
		// Positions
        vector<Atom> &atoms = state->atoms;
        for (int i=0; i<atoms.size(); i++) {
            Atom &a = atoms[i];
            a.pos = pos[i];
            a.vel = vel[i];
            a.force = force[i];
            a.id = ids[i];
            a.type = types[i];
		}

		// LAMMPS computes will reset thermo data based on
		// updated information. No need to sync thermo data
		// from snapshot to engine.
		// However, this will change in the future.
	}
}
