#pragma once 

#include "CollectiveVariable.h"

#include <array>
#include <math.h>
#include "GPUArrayDeviceGlobal.h"
namespace SSAGES
{
	// Collective variable on an atom coordinate. This will
	// return the value of either the x, y, or z coordinate
	// depending on the user specification for a defined atom.
	class AtomCoordinateCV_gpu : public CollectiveVariable
	{
	private:
		// ID of atom of interest.
		int _atomid; 

		// Index of dimension. 0 -> x, 1 -> y, 2 -> z.
		int _index;

		// pointer to cvValArray + myIdxInCVList
		float *_val;

		// Gradient vector dOP/dxi.
        //
        GPUArrayDeviceGlobal<float4> _grad;
        std::vector<Vector3> junkForCompliance;
        std::array<double, 2> moreJunkForCompliance;


	public:
		// Construct an atom coordinate CV. The atomid specifies the 
		// ID of the atom of interest, and index specifies the dimension 
		// to report with 0 -> x, 1 -> y, 2 -> z. 
		// TODO: bounds needs to be an input.
		AtomCoordinateCV_gpu(int atomid, int index) :
		_atomid(atomid), _index(index)
		{
		}

		// Initialize necessary variables.
		void Initialize(const Snapshot& snapshot) override
		{
			// Initialize gradient. 
			auto n = snapshot.GetPositions().size();
            _grad = GPUArrayDeviceGlobal<float4>(n);
		}
        void takeValPtr(float *val) {
            _val = val;
        }

		// Evaluate the CV.
		void Evaluate(const Snapshot& snapshot) override;
	

		// Return the value of the CV.
		double GetValue() const override 
		{ 
			//return _val; 
            return 0;
		}

		double GetPeriodicValue(double Location) const override
		{
			return Location;
		}

		// Return the gradient of the CV.
		const std::vector<Vector3>& GetGradient() const override
		{
            return junkForCompliance;
		}

		// Return the boundaries of the CV.
		const std::array<double, 2>& GetBoundaries() const override
		{
			return moreJunkForCompliance;
		}

		double GetDifference(const double Location) const override
		{
            return 0;
		}
	};
}
