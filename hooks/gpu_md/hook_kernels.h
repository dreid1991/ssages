#include "cutils_math.h"

void copyToBuffer(float4 *xs, float4 *vs, uint *ids, int *idToIdxs, char *buffer, uint *idsToCopy, int n);


void unpackBuffer(float4 *fs, int *idToIdxs, float4 *biasForces, uint *idsToCopy, int n);
