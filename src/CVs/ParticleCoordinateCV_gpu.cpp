#include "ParticleCoordinateCV_gpu.h"
#include "ParticleCoordinateCV_gpu_kernels.h"

using namespace SSAGES;

void ParticleCoordinateCV_gpu::Initialize(const Snapshot& snapshot)
{
    using std::to_string;

    auto n = _atomids.size();

    int atomsize = snapshot._gpd.nAtoms;
    d_grad = GPUArrayDeviceGlobal<float4>(atomsize);
    d_reduction_buf = GPUArrayDeviceGlobal<float4>(n);            
    d_sum_buf = GPUArrayDeviceGlobal<float4>(1);            

    d_val = GPUArrayGlobal<float>(1);            
//     d_mass = GPUArrayGlobal<float>(1);            

    //DANMD doesn't support MPI for now
    //but atomids should be copied to device
    d_atomids = GPUArrayGlobal<int>(n);
    for(size_t i = 0; i < n; ++i)
    {
        d_atomids.h_data[i]=int(_atomids[i]);
    }
    d_atomids.dataToDevice();            
    
/*			// Make sure atom ID's are on at least one processor. 
    std::vector<int> found(n, 0);
    for(size_t i = 0; i < n; ++i)
    {
        if(snapshot.GetLocalIndex(_atomids[i]) != -1)
            found[i] = 1;
    }

    MPI_Allreduce(MPI_IN_PLACE, found.data(), n, MPI_INT, MPI_SUM, snapshot.GetCommunicator());
    unsigned ntot = std::accumulate(found.begin(), found.end(), 0, std::plus<int>());
    if(ntot != n)
        throw BuildException({
            "ParticleCoordinateCV: Expected to find " + 
            to_string(n) + 
            " atoms, but only found " + 
            to_string(ntot) + "."
        });	*/		
}


void ParticleCoordinateCV_gpu::Evaluate(const Snapshot& snapshot) 
{
    // Gradient and value. 
    float4 *xs = snapshot._gpd.xs;
    float4 *vs = snapshot._gpd.vs;
    
    int *idToIdxs = snapshot._gpd.idToIdxs;
    int nAtoms = snapshot._gpd.nAtoms;
    
    BoundsGPU boundsGPU = snapshot._gpd.boundsGPU;

//     d_reduction_buf.memset(0.0f);//TODO should not be necessary
//     d_val.memsetByVal(0.0f);
    
    d_sum_buf.memsetByVal(make_float4(0, 0, 0, 0));
    float3 index = make_float3(0,0,0);
    switch(_dim)
	{
        case Dimension::x:
        index.x=1;
        break;
        case Dimension::y:
        index.y=1;
        break;
        case Dimension::z:
        index.z=1;
        break;
    }
    call_particle_position_mass_eval(xs,vs,boundsGPU,idToIdxs,d_reduction_buf.data(),d_sum_buf.data(),d_val.getDevData(), d_atomids.getDevData(),_atomids.size(),index, nAtoms, snapshot._gpd.warpSize);
    
    d_grad.memset(0.0f);
    
    call_grad_eval(vs,idToIdxs,d_grad.data(), d_atomids.getDevData(),_atomids.size(),d_sum_buf.data(),index);
       
}

double ParticleCoordinateCV_gpu::GetValue()
{
    d_val.dataToHost();
    _val=d_val.h_data[0];
//       printf("GetValue   %f \n", _val);

    return _val;
}