/**
 * This file is part of
 * SSAGES - Suite for Advanced Generalized Ensemble Simulations
 *
 * Copyright 2016 Hythem Sidky <hsidky@nd.edu>
 *
 * SSAGES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SSAGES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SSAGES.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once 

#include "Drivers/DriverException.h"
#include "CollectiveVariable.h"
#include "GPUArrayGlobal.h"

namespace SSAGES
{
	//! Collective variable on a particle coordinate. 
	/*!
	 * This will return the value of either the x, y, or z coordinate, depending
	 * on the user specification for a defined particle, which is a collection of 
	 * one or more atoms.
	 *
	 * \ingroup CVs
	 */
	class ParticleCoordinateCV_gpu : public CollectiveVariable
	{
	private:
		//! IDs of atoms of interest. 
	 	Label _atomids; 

	 	//! Index of dimnesion.
	 	Dimension _dim;

        GPUArrayDeviceGlobal<float4> d_grad;
        GPUArrayDeviceGlobal<float4> d_reduction_buf;//buffer for parallel reduction 
        GPUArrayDeviceGlobal<float4> d_sum_buf;//buffer for parallel reduction 

        GPUArrayGlobal<float> d_val; 
//         GPUArrayGlobal<float> d_mass; 
        GPUArrayGlobal<int> d_atomids;
        
	public:
		//! Constructor.
		/*!
		 * \param atomids Atom ID's of interest.
		 * \param index Index of dimension.
		 *
		 * Construct a particle coordinate CV. The atomids specify a vector of the atom
		 * ID's of interest, and index specifies the dimension to report (x,y,z).
		 *
		 * \todo Bounds needs to be an input.
		 */	 	
		ParticleCoordinateCV_gpu(const Label& atomids, Dimension dim) : 
		_atomids(atomids), _dim(dim)
		{}

		//! Initialize necessary variables.
		/*!
		 * \param snapshot Current simulation snapshot.
		 */
		void Initialize(const Snapshot& snapshot) override;


		//! Evaluate the CV.
		/*!
		 * \param snapshot Current simulation snapshot.
		 */
		void Evaluate(const Snapshot& snapshot) override;

		//! Return value taking periodic boundary conditions into account.
		/*!
		 * \param Location Get wrapped value of this location.
		 *
		 * \return Input value
		 *
		 * The AtomCoordinate CV does not consider periodic boundary
		 * conditions. Thus, this function always returns the input value.
		 */
        
		//! Get current value of the CV.
		/*!
		 * \return Current value of the CV.
		 *
		 * Returns the current value of the CV which has been computed before
		 * via the call to CollectiveVariable::Evaluate().
		 */
		double GetValue() override;
		//! Get current gradient of the CV.
		/*!
		 * \return Per-atom gradient of the CV.
		 *
		 * Returns the current value of the CV gradient. This should be an n
		 * length vector, where n is the number of atoms in the snapshot. Each
		 * element in the vector is the derivative of the CV with respect to
		 * the atom's coordinate (dCV/dxi).
		 */
		const std::vector<Vector3>& GetGradient() const 
		{
          throw BuildException({"ParticleCoordinateCV:grad is not accessible on CPU"});
			return _grad;
		}
        
		double GetPeriodicValue(double Location) const override
		{
			return Location;
		}

		//! Return difference considering periodic boundaries.
		/*!
		 * \param Location Calculate difference of CV value to this location.
		 *
		 * \return Direct difference.
		 *
		 * As the AtomCoordinate CV does not consider periodic boundary
		 * conditions, the difference between the current value of the CV and
		 * another value is always the direct difference.
		 */
		double GetDifference(const double Location) const override
		{
			return _val - Location;
		}

		//! Serialize this CV for restart purposes.
		/*!
		 * \param json JSON value
		 */
		void Serialize(Json::Value& json) const override
		{
			json["type"] = "ParticleCoordinate";			
			switch(_dim)
			{
				case Dimension::x:
					json["dimension"] = "x";
					break;
				case Dimension::y:
					json["dimension"] = "y";
					break;
				case Dimension::z:
					json["dimension"] = "z";
					break;
			}
			
			for(auto& id : _atomids)
				json["atom_ids"].append(id);

			for(auto& bound : _bounds)
				json["bounds"].append(bound);
		}
        float4 *GetGradient_gpu() {
            return d_grad.data();
        }
        float *GetValue_gpu() {
            return d_val.getDevData();
        }
	 };
}