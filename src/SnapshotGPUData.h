#pragma once
#include "GPUArrayDeviceGlobal.h"
namespace SSAGES{
    class SnapshotGPUData {
        private:
            void processXs(){};//to be defined be engine for each field with nonconforming format
            void processVs(){};
            void processFs(){};
            void processIds(){};
        public:
            SnapshotGPUData() {
                //not dealing with possibility of multiple formats currently
                xsFormatSame = vsFormatSame = fsFormatSame = idsFormatSame = true;
            }
            int nAtoms;
            float4 *xs;
            float4 *vs;
            float4 *fs;
            uint *ids;
            int *idToIdxs;


            GPUArrayDeviceGlobal<float> cvValues;
            bool xsFormatSame, vsFormatSame, fsFormatSame, idsFormatSame;

            bool xsIsProcessed, vsIsProcesses, fsIsProcessed, idsIsProcessed;
            //only to be used by fields for which format is not as expected by package
          //  GPUArrayDeviceGlobal<float4> xsProcessed;
          //  GPUArrayDeviceGlobal<float4> vsProcessed;
          //  GPUArrayDeviceGlobal<float4> fsProcessed;
          //  GPUArrayDeviceGlobal<int> idsProcessed;
          //  GPUArrayDeviceGlobal<int> typesProcessed;
            void resetIsProcessed() {
               // xsIsProcessed = false;
               // vsIsProcessed = false;
                //fsIsProcessed = false;
                //idsIsProcessed = false;
            }
            /*
            float4 *GetPositions () {
                if (xsFormatSame) {
                    return xs;
                } else {
                    if (!xsIsProcessed) {
                        processXs();
                        xsIsProcessed = true;
                    }
                    return xsProcessed.ptr;
                }
            }
            float4 *GetVelocities () {
                if (vsFormatSame) {
                    return vs;
                } else {
                    if (!vsIsProcessed) {
                        processVs();
                        vsIsProcessed = true;
                    }
                    return vsProcessed.ptr;
                }
            }
            float4 *GetForces () {
                if (fsFormatSame) {
                    return fs;
                } else {
                    if (!fsIsProcessed) {
                        processFs();
                        fsIsProcessed = true;
                    }
                    return fsProcessed.ptr;
                }
            }
            int *GetAtomIDs () {
                if (idsFormatSame) {
                    return ids;
                } else {
                    if (!idsIsProcessed) {
                        processIds();
                        idsIsProcessed = true;
                    }
                    return idsProcessed.ptr;
                }
            }
            */

    };
}
