/**
 * This file is part of
 * SSAGES - Suite for Advanced Generalized Ensemble Simulations
 *
 * Copyright 2016 Hythem Sidky <hsidky@nd.edu>
 *                Ben Sikora <bsikora906@gmail.com>
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

#include "../Snapshot.h"
#include "../JSON/Serializable.h"
#include "types.h"
#include <vector>

// Forward declare.
namespace Json {
	class Value;
}

namespace SSAGES
{
	//! Abstract class for a collective variable.
	/*!
	 * \ingroup CVs
	 */
	class CollectiveVariable: public Serializable
	{
	protected:
		//! Gradient vector dCv/dxi.
		std::vector<Vector3> _grad;
		
	 	//! Current value of CV.
		double _val;				

		// Bounds on CV.
		std::array<double, 2> _bounds;		
	public:
		//! Constructor.
		CollectiveVariable() : 
		_grad(0), _val(0), _bounds{{0,0}}
		{}

		//! Destructor.
		virtual ~CollectiveVariable(){}

		//! Initialize CV.
		/*!
		 * Initialization of the CV. This is an optional method and is called
		 * during the pre-simulation phase of the hook. It is typically used to
		 * allocate/reserve memory.
		 */
		virtual void Initialize(const Snapshot&) {}

		//! Evaluate CV.
		/*!
		 * \param snapshot Reference of the current simulation snapshot.
		 *
		 * Evaluation of the CV. This function is called by the Hook in the
		 * post-integration phase every iteration. The CV should compute its
		 * value and gradient, storing them in a local private variable.
		 */
		virtual void Evaluate(const Snapshot& snapshot) = 0;

		//! Get current value of the CV.
		/*!
		 * \return Current value of the CV.
		 *
		 * Returns the current value of the CV which has been computed before
		 * via the call to CollectiveVariable::Evaluate().
		 */
		virtual double GetValue() /*const*/
		{
			return _val;
		}

        //! Returns the minimum image of a CV based on the input location.
        /*!
		 * \param Value against which the minimum image is calculated.
		 * \return Minimum image of the CV 
		 *
         * Takes the input location and applies the periodic boundary conditions to return a minimum image
         * of the CV.
		 */
		virtual double GetMinimumImage(double location) const
		{
			return _val;
		}

		//! Apply periodic boundaries to a given value.
		/*!
		 * \param location Value to which the periodic boundaries should be applied.
		 * \return Correct value.
		 *
		 * Takes location and applies periodic boundaries of the CV on it and
		 * returns a correct value. Example would be torsional angle which has
		 * bounds at pi and -pi. If location = 2pi, GetPeriodicValue(location)
		 * would return 0.
		 */
		virtual double GetPeriodicValue(double location) const
		{
			return location;
		}

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
			return _grad;
		}

		//! Get CV boundaries.
		/*!
		 * \return List of lower and upper boundaries of the CV.
		 *
		 * Returns the boundaries of the CV. These represent the bounds within
		 * which the CV is expected to be constrained. There is no requirement
		 * on the method to respect the values returned here.
		 */
		const std::array<double, 2>& GetBoundaries()
		{
			return _bounds;
		}

		//! Get difference between current CV value and a given value, taking
		//! periodic boundaries into account.
		/*!
		 * \param Location Value whose distance from the current CV value should be calculated.
		 * \return Difference taking periodic boundaries into account.
		 *
		 * Returns the difference betwen the current cv value and Location:
		 * (_value - Location) respecting periodic boundary conditions of the
		 * CV, if the CV has periodic boundary conditions. For example Torsional
		 * angle has boundaries at pi and -pi, in which the difference beteen
		 * the angles is 0 not 2pi
		 */
		virtual double GetDifference(double location) const
		{
			return _val - location;
		}

        //! Returns the ids used in this collective variable
        /*!
         * \return std::vector of all atom ids involved in this collective variable
         */
        virtual std::vector<int> getIds() {
            return std::vector<int>(); //to do: make this purely virtual so all CVs must implement.  Currently just doing testing
        }; 
		//! Set up collective variable.
		/*!
		 * \param json JSON input value.
		 * \return Pointer to the CV built by this function. \c nullptr in case of unknown error.
		 *
		 * Builds a CV from a JSON node. Returns a pointer to the built cv. If
		 * an unknown error is encountered, this function will return a
		 * \c nullptr, but generally it will throw a BuildException on failure.
		 * \warning Object lifetime is the caller's responsibility.
		 */
		static CollectiveVariable* BuildCV(const Json::Value& json);

		//! Set up collective variable.
		/*!
		 * \param json JSON input value.
		 * \param path Path for JSON path specification.
		 * \return Pointer to the CV built by this function. \c nullptr in case of unknown error.
		 *
		 * This function overloads CollectiveVariable::BuildCV(const Json::Value&).
		 */
		static CollectiveVariable* BuildCV(const Json::Value& json, 
							   const std::string& path);

		//! Set up CV and add it to list of CVs.
		/*!
		 * \param json JSON input value.
		 * \param cvlist List of CVs to which the new CV will added.
		 * \param path Path for JSON path specification.
		 *
		 * Builds CVs and adds them to the List of CV.
		 */
		static void BuildCV(const Json::Value& json, 
							   CVList& cvlist,
							   const std::string& path);
	};
}
