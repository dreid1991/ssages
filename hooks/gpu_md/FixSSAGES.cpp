#include <iostream>

#include "FixSSAGES.h"
#include "State.h"
#include <set>
#include "hook_kernels.h"
//HEY - YOU NEED TO ADD force_last AS ONE OF THE PER-ATOM VARIABLES IF YOU WANT TO BE ABLE TO FULLY TRANSFER STATES
using namespace SSAGES;
namespace SSAGES
{
	FixSSAGES::FixSSAGES(SHARED(State) state, std::string handle) :
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

	bool FixSSAGES::prepareForRun()
	{
        //so this is always called while data is on cpu
		// Allocate vectors for snapshot.


        std::set<int> allActiveIds;
        for(auto& cv : _cvs) {
            std::vector<int> ids = cv->getIds();
            for (int id : ids) {
                allActiveIds.insert(id);
            }
        }
        std::vector<uint> activeIds;
        for (int x : allActiveIds) {
            activeIds.push_back(x);
            printf("ACTIVE ID %d\n", x);
        }
        _activeIds = GPUArrayGlobal<uint>(activeIds);
        _activeIds.dataToDevice();
        //need to copy off xs, vs, fs, ids
        int n = _activeIds.size();

        int totalBufferSize = (sizeof(float4) + sizeof(float4) + sizeof(float4) + sizeof(int)) * n;

        _dataBuffer = GPUArrayGlobal<char>(totalBufferSize);
        _forceBuffer = GPUArrayGlobal<float4>(n);


		_snapshot->SetNumAtoms(n);

		auto& pos = _snapshot->GetPositions();
		pos.resize(n);
		auto& vel = _snapshot->GetVelocities();
		vel.resize(n);
		auto& frc = _snapshot->GetForces();
		frc.resize(n);
		auto& masses = _snapshot->GetMasses();
		masses.resize(n);
		auto& ids = _snapshot->GetAtomIDs();
		ids.resize(n);
		auto& types = _snapshot->GetAtomTypes();
		types.resize(n);

        //ignoring charges for now. Will need them when config-reset stuff is implemented
		SyncToSnapshot();
        /*
		Hook::PreSimulationHook();


        */
        return true;
	}

	void FixSSAGES::compute(bool computeVirials)
	{
        SyncToSnapshot();
		Hook::PostIntegrationHook();
	}

	bool FixSSAGES::postRun()
	{
		SyncToSnapshot();
		Hook::PostSimulationHook();
        return true;
	}
    void FixSSAGES::SyncToSnapshot() {
        BoundsGPU bounds = state->boundsGPU;
		Matrix3 H;
		H << bounds.rectComponents.x, 0, 0, 
		                0, bounds.rectComponents.y, 0,
		                0,            0, bounds.rectComponents.z;

		_snapshot->SetHMatrix(H);

		// Get box origin. 
		Vector3 origin;
		origin = {
			bounds.lo.x,
			bounds.lo.y,
			bounds.lo.z
		};
	
		_snapshot->SetOrigin(origin);

		// Set periodicity. 
		_snapshot->SetPeriodicity({
			bounds.periodic.x,
			bounds.periodic.y,
			bounds.periodic.z
		});





        int n = _snapshot->GetNumAtoms();
        copyToBuffer(state->gpd.xs.getDevData(), state->gpd.vs.getDevData(), state->gpd.ids.getDevData(), state->gpd.idToIdxs.d_data.data(), _dataBuffer.d_data.data(), _activeIds.getDevData(), n);
        _dataBuffer.dataToHost();
        cudaDeviceSynchronize();
        auto& pos = _snapshot->GetPositions();
		auto& vel = _snapshot->GetVelocities();
		auto& frc = _snapshot->GetForces();
		auto& masses = _snapshot->GetMasses();
		auto& ids = _snapshot->GetAtomIDs();
		auto& types = _snapshot->GetAtomTypes();


        float4 *posesPreproc = (float4 *) _dataBuffer.h_data.data();
        float4 *velsPreproc = ((float4 *) _dataBuffer.h_data.data()) + n;
        uint *idsPreproc = (uint *) (((float4 *) _dataBuffer.h_data.data()) + 2 * n);
        //float4 *posesPreproc = _dataBuffer.h_data.begin();
        for (int i=0; i<n; i++) {
            float4 posPre = posesPreproc[i];
            float4 velPre = velsPreproc[i];
            uint idPre = idsPreproc[i];
            int type = * (int *) &posPre.w;
            double mass = 1.0 / velPre.w;
            pos[i][0] = posPre.x;
            pos[i][1] = posPre.y;
            pos[i][2] = posPre.z;

            vel[i][0] = velPre.x;
            vel[i][1] = velPre.y;
            vel[i][2] = velPre.z;

            frc[i][0] = 0; //ASSUMING BIASING FORCES DO NOT DEPEND ON CURRENT FORCE
            frc[i][1] = 0;
            frc[i][2] = 0;

            masses[i] = mass;
            types[i] = type;
            ids[i] = idPre;
        }

        
    }
    void FixSSAGES::SyncToEngine() {
        int n = _snapshot->GetNumAtoms();
		auto& frc = _snapshot->GetForces();
        for (int i=0; i<n; i++) {
            _forceBuffer.h_data[i].x = frc[i][0];
            _forceBuffer.h_data[i].y = frc[i][1];
            _forceBuffer.h_data[i].z = frc[i][2];
        }
        _forceBuffer.dataToDevice();

        unpackBuffer(state->gpd.fs.getDevData(), state->gpd.idToIdxs.d_data.data(), _forceBuffer.d_data.data(), _activeIds.getDevData(), n);



    }
}
