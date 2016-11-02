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
#include "Umbrella.h"

#include <iostream>

namespace SSAGES
{
	//! Value of the harmonic potential. Helper function.
	/*!
	 * \param k Spring constant.
	 * \param x0 Equilibrium extension.
	 * \param x Current extension.
	 * \return Harmonic potential at extension x.
	 */
	double spring(double k, double x0, double x)
	{
		return 0.5 * k * (x - x0) * (x - x0);
	}

	void Umbrella::PreSimulation(Snapshot*, const CVList& cvs)
	{
		if(_comm.rank() == 0)
		{
		 	_umbrella.open(_filename.c_str(), std::ofstream::out | std::ofstream::app);
		 }
	}

	void Umbrella::PostIntegration(Snapshot* snapshot, const CVList& cvs)
	{
		// Compute the forces on the atoms from the CV's using the chain 
		// rule.
		auto& forces = snapshot->GetForces();
		for(size_t i = 0; i < cvs.size(); ++i)
		{
			// Get current CV and gradient.
			auto& cv = cvs[i];
			auto& grad = cv->GetGradient();

			// Compute dV/dCV.
			auto D = _kspring[i]*(cv->GetDifference(_centers[i]));

			// Update forces.
			for(size_t j = 0; j < forces.size(); ++j)
				for(size_t k = 0; k < forces[j].size(); ++k)
					forces[j][k] -= D*grad[j][k];
		}

		_iteration++;
		if(_iteration % _logevery == 0)
			PrintUmbrella(cvs);
	}

	void Umbrella::PostSimulation(Snapshot*, const CVList&)
	{
		if(_comm.rank() ==0)
		{
			_umbrella.close();
		}
	}

	void Umbrella::PrintUmbrella(const CVList& CV)
	{
		if(_comm.rank() ==0)
		{
			_umbrella.precision(8);
			_umbrella << _iteration << " ";

			for(size_t jj = 0; jj < _centers.size(); jj++)
				_umbrella<< _centers[jj] << " " << CV[jj]->GetValue()<< " "; 

			_umbrella<<std::endl;
		}
	}
}