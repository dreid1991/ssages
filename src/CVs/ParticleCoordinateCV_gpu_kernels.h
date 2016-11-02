#include "BoundsGPU.h"


void call_particle_position_mass_eval(float4 *xs,float4 *vs,BoundsGPU boundsGPU, int *idToIdxs, float4 *buf,float4 *sum_buf,float* val, int* atomIds,int atomIdsize, float3 index, int nAtoms, int warpize);


void call_grad_eval(float4 *vs, int *idToIdxs, float4 *grad, int* atomIds,int atomIdsize,float4 *totalmass, float3 index);


// void call_particle_coordinate_eval(float4 *xs,float4 *vs, BoundsGPU boundsGPU, int *idToIdxs, float *buf, float4 *grad, int* atomIds, int atomIdsize, float *val,float massinv,int index, int nAtoms, int warpSize);
// 
// 
// void call_particle_mass_eval(float4 *vs, int *idToIdxs, float *buf,int* atomIds, int atomIdsize, float *mass, int nAtoms, int warpSize);
// 
// void call_wrap_particle_coordinate_eval(BoundsGPU boundsGPU, float *val,float3 index);