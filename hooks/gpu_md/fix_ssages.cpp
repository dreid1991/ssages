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
	}

	void FixSSAGES::compute()
	{
        syncToSnapshot();
		Hook::PostIntegrationHook();
	}

	void FixSSAGES::postRun()
	{
        //data is back on cpu at this point
		SyncToSnapshotCPU();
		Hook::PostSimulationHook();
	}
    void FixSSAGES::syncToSnapshot() {
        GPUData &gpd = state->gpd;
        _snapshot._gpd.xs = gpd.xs.getDevData()
        _snapshot._gpd.vs = gpd.vs.getDevData()
        _snapshot._gpd.fs = gpd.fs.getDevData()
        _snapshot._gpd.ids = gpd.ids.getDevData()
    }
    void FixSSAGES::syncToEngine() {
        //nothing here - operating on the same list on the gpu    
    }
	void FixSSAGES::SyncToSnapshotCPU()
	{
        //NEED TO ADD GROUPTAG TO SSAGES SNAPSHOT AND COPY IT ON/OFF OF GPU
        GPUData &gpd = state->gpd;
        gpd.xs.dataToHost();
        gpd.vs.dataToHost();
        gpd.fs.dataToHost();
        gpd.ids.dataToHost();
        cudaDeviceSynchronize();
        //this is only called when data in cpu, so between runs (like when exchanging walkers or something)	

		auto& pos = _snapshot.GetPositions();
		auto& vel = _snapshot.GetVelocities();
		auto& frc = _snapshot.GetForces();

		// Labels and ids for future work on only updating
		// atoms that have changed.
		auto& ids = _snapshot.GetAtomIDs();
		auto& types = _snapshot.GetAtomTypes();
        std::vector<double> _snapshot.GetAtomMasses();

		// Get iteration.
		_snapshot.SetIteration(state->turn);
		
		// Get volume.
		double vol = state->bounds.volume();
		_snapshot.SetVolume(vol);
        //set temperature, eng?

        int nAtoms = state->atoms.size();
        for (int i=0; i<nAtoms; i++) {
            pos[i] = Vector3(gpd.xs.h_data[i].x, gpd.xs.h_data[i].y, gpd.xs.h_data[i].z);
            //same for the rest
            vel[i] = atoms.vel;
            frc[i] = atom.force;

            ids[i] = atom.id;
            types[i] = atom.type; 
        }
		// Update values.
        
			
    }

	void FixSSAGES::SyncToEngineCPU() //put Snapshot values -> LAMMPS
	{
		// Obtain local const reference to snapshot variables.
		// Const will ensure that _snapshot variables are
		// not being changed. Only engine side variables should
		// change. 
        //
        //HEY - MAKE IT CHECK TO BUILD NEIGHBORLISTS THIS TURN BECAUSE YOU ARE POTENTIALLY SCRAMBLING YOUR ATOMS
		const auto& pos = _snapshot.GetPositions();
		const auto& vel = _snapshot.GetVelocities();
		const auto& frc = _snapshot.GetForces();

		// Labels and ids for future work on only updating
		// atoms that have changed.
		const auto& ids = _snapshot.GetAtomIDs();
		const auto& types = _snapshot.GetAtomTypes();
        std::vector<double> masses = _snapshot.GetAtomMasses();

        GPUData &gpd = state->gpd;

        int nAtoms = state->atoms.size();
        for (int i=0; i<atoms.size(); i++) {
            gpd.xs.h_data[i] = make_float4(pos[i][0], pos[i][1], pos[i][2], * (float *)&types[i]);
            gpd.vs.h_data[i] = make_float4(vel[i][0], vel[i][1], vel[i][2], 1/masses[i]);
		}
        gpd.xs.dataToDevice();
        gpd.vs.dataToDevice();
        gpd.fs.dataToDevice();
        gpd.ids.dataToDevice();
        state->gridGPU.perBoundaryConditions();


		// LAMMPS computes will reset thermo data based on
		// updated information. No need to sync thermo data
		// from snapshot to engine.
		// However, this will change in the future.
	}
}
