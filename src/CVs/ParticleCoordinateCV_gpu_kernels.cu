#include "ParticleCoordinateCV_gpu_kernels.h"
#include "globalDefs.h"//from DANMD
#include "cutils_func.h"


__global__ void call_particle_position_mass_eval_cu(float4 *xs,float4 *vs, BoundsGPU bounds, int *idToIdxs, float4 *buf, int* atomIds,int atomIdsize, int nAtoms) {
    int idx = GETIDX();
    if (idx < atomIdsize) {
        int id = atomIds[idx];
        int Idx= idToIdxs[id];
        float4 x=xs[Idx];
        float3 pos = make_float3(x);
        //get first atom postion
        float3 pos0 = make_float3(xs[idToIdxs[atomIds[0]]]);
        float3 dr = bounds.minImage(pos - pos0)+pos0;
        float massinv=vs[Idx].w;
        x=make_float4(dr);
        x.w=massinv;
        buf[idx] = x;
        
    }
}

__global__ void call_grad_eval_cu(float4 *vs, int *idToIdxs, float4 *grad, int* atomIds,int atomIdsize,float4 *totalmass, float3 index) {
    int idx = GETIDX();
    if (idx < atomIdsize) {
        int id = atomIds[idx];
        int Idx= idToIdxs[id];
        float massinv=vs[Idx].w;
        float4 atomGrad = make_float4(index)/(massinv*totalmass[0].w);
        grad[Idx] = atomGrad;
//          printf("atomGrad  %f %f %f  %f %d\n", atomGrad.x,atomGrad.y,atomGrad.z,atomGrad.w,id);

    }
}

__global__ void call_wrap_particle_coordinate_eval_cu(BoundsGPU bounds, float *val,float4 *sum_buf, float3 index) {
    int idx = GETIDX();
    if (idx < 1) {
        float4 posmass=sum_buf[0];
        float3 pos=make_float3(posmass)/posmass.w;

//         pos = bounds.wrap(pos);
        float3 trace = bounds.trace();
        float3 diffFromLo = pos - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        pos -= trace * imgs * bounds.periodic;
        float a=dot(pos,index);

        val[0]=a;

    }
}



void call_particle_position_mass_eval(float4 *xs,float4 *vs,BoundsGPU boundsGPU, int *idToIdxs, float4 *buf,float4 *sum_buf,float* val, int* atomIds,int atomIdsize, float3 index, int nAtoms, int warpize)
{
  

    SAFECALL((call_particle_position_mass_eval_cu<<<NBLOCK(atomIdsize), PERBLOCK>>>(xs, vs,boundsGPU,idToIdxs, buf, atomIds, atomIdsize, nAtoms)));

    SAFECALL((accumulate_gpu<float4,float4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(atomIdsize/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float4)*PERBLOCK>>>
            (
             sum_buf,
             buf,
             atomIdsize,
             warpize,
             SumVectorXYZOverW()
             )));    
    
     SAFECALL((call_wrap_particle_coordinate_eval_cu<<<NBLOCK(1), PERBLOCK>>>(boundsGPU,val, sum_buf,index)));
}


 void call_grad_eval(float4 *vs, int *idToIdxs, float4 *grad, int* atomIds,int atomIdsize,float4 *totalmass, float3 index) {
      SAFECALL((call_grad_eval_cu<<<NBLOCK(atomIdsize), PERBLOCK>>>( vs,idToIdxs, grad, atomIds, atomIdsize, totalmass,index)));
 }


__global__ void call_particle_coordinate_eval_cu(float4 *xs,float4 *vs, BoundsGPU bounds, int *idToIdxs, float *buf, float4 *grad, int* atomIds,int atomIdsize,float massinv, int index, int nAtoms) {
    int idx = GETIDX();
    if (idx < atomIdsize) {
        int id = atomIds[idx];
        int Idx= idToIdxs[id];
        float4 atomGrad = make_float4(0, 0, 0, 0);
        float tval=0;
        float4 x=xs[Idx];
        float3 pos = make_float3(x);
        //get first atom postion
        float3 pos0 = make_float3(xs[idToIdxs[atomIds[0]]]);
        float3 dr = bounds.minImage(pos - pos0)+pos0;
        float4 v=vs[Idx];
        if (index == 0) {
            atomGrad.x = 1;
            tval = dr.x;
        } else if (index == 1) {
            atomGrad.y = 1;
            tval = dr.y;
        } else if (index == 2) {
            atomGrad.z = 1;
            tval = dr.z;
        }
        grad[Idx] = atomGrad/v.w*massinv;
        buf[idx] = tval/v.w*massinv;//TODO use buf value or do not calc mass every timestep
//  printf("buf   %f %d\n", tval,id);
    }
}


void call_particle_coordinate_eval(float4 *xs,float4 *vs,BoundsGPU boundsGPU, int *idToIdxs, float *buf, float4 *grad, int* atomIds,int atomIdsize,float *val,float massinv, int index, int nAtoms, int warpize) {
  

    SAFECALL((call_particle_coordinate_eval_cu<<<NBLOCK(atomIdsize), PERBLOCK>>>(xs, vs,boundsGPU,idToIdxs, buf, grad, atomIds, atomIdsize,massinv, index, nAtoms)));

    SAFECALL((accumulate_gpu<float,float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(atomIdsize/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
            (
             val,
             buf,
             atomIdsize,
             warpize,
             SumSingle()
             )));     
}


__global__ void call_particle_mass_eval_cu(float4 *vs, int *idToIdxs, float *buf,  int* atomIds,int atomIdsize, int nAtoms) {
    int idx = GETIDX();
    if (idx < atomIdsize) {
        int id = atomIds[idx];
        int Idx= idToIdxs[id];
        buf[idx] = 1.0/vs[Idx].w;
    }
}


void call_particle_mass_eval(float4 *vs, int *idToIdxs, float *buf,int* atomIds, int atomIdsize, float *mass, int nAtoms, int warpSize) {

    SAFECALL((call_particle_mass_eval_cu<<<NBLOCK(atomIdsize), PERBLOCK>>>(vs, idToIdxs, buf,  atomIds, atomIdsize, nAtoms)));
    
    SAFECALL((accumulate_gpu<float,float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(atomIdsize/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
            (
             mass,
             buf,
             atomIdsize,
             warpSize,
             SumSingle()
             )));     

}


__global__ void call_wrap_particle_coordinate_eval_cu(BoundsGPU bounds, float *val, float3 index) {
    int idx = GETIDX();
    if (idx < 1) {
        float a=val[0];
        float3 pos = index*a;
//         pos = bounds.wrap(pos);
        float3 trace = bounds.trace();
        float3 diffFromLo = pos - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        pos -= trace * imgs * bounds.periodic;
        if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
            a=dot(pos,index);
            val[0]=a;
        }
    }
}


void call_wrap_particle_coordinate_eval( BoundsGPU boundsGPU, float *val, float3 index){

    
      SAFECALL((call_wrap_particle_coordinate_eval_cu<<<NBLOCK(1), PERBLOCK>>>(boundsGPU,val, index)));
  
}
