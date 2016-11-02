/**
 * This file is part of
 * SSAGES - Suite for Advanced Generalized Ensemble Simulations
 *
 * Copyright 2016 Ben Sikora <bsikora906@gmail.com>
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

#include "StringMethod.h"
#include "../CVs/CollectiveVariable.h"
#include <fstream>

namespace SSAGES
{
	//! Multi-walker Elastic Band.
	/*!
	 * Implementation of a multi-walker Elastic Band method with no bells and
	 * whistles.
	 *
	 * \ingroup Methods
	 */
	class ElasticBand : public StringMethod
	{
	private:	

		//! Number Equilibration steps, number of MD steps to
		//! allow the system to reequilibrate before evolving.
		unsigned int _equilibrate;

		//! Number evolution steps, number of MD steps before
		//! collecting statistics for gradients.
		unsigned int _evolution;

		//! Block iterations
		unsigned int _nsamples;

		//! Number samples actually sampled.
		unsigned int _nsampled;

		//! Time step of string change
		double _tau;

		//! String spring constant.
		double _stringspring;

		//! Updates the nudged elastic band string.
		void StringUpdate();

	public:
		//! Constructor
		/*!
		 * \param world MPI global communicator.
		 * \param comm MPI local communicator.
		 * \param centers List of centers.
		 * \param maxiterator Maximum number of iterations.
		 * \param blockiterations Number of samples to collect before updating string.
		 * \param tau Value of tau (default: 0.1).
		 * \param cvspring Spring constants for cvs.
		 * \param equilibrate Number of MD steps to allow the system to reequilibrate after updating string.
		 * \param evolution Number of MD steps to allow the system to evolve before gathering statistics.
		 * \param stringspring Spring constant used between nodes.
		 * \param frequency Frequency with which this method is invoked.
		 *
		 * Constructs an instance of Elastic Band method.
		 */
		ElasticBand(boost::mpi::communicator& world,
					boost::mpi::communicator& comm,
					const std::vector<double>& centers,
					unsigned int maxiterations,
					unsigned int nsamples,
					double tau,
					const std::vector<double> cvspring,
					unsigned int equilibrate,
					unsigned int evolution,
					double stringspring,
			 		unsigned int frequency) : 
		StringMethod(world, comm, centers, maxiterations,
		cvspring, frequency), _equilibrate(equilibrate),
		_evolution(evolution), _nsamples(nsamples),
		_nsampled(0), _tau(tau), _stringspring(stringspring)
		{
		}

		//! Post-integration hook.
		/*!
		 * \param snapshot Current simulation snapshot.
		 * \param cvs List of CVs.
		 */
		void PostIntegration(Snapshot* snapshot, const CVList& cvs) override;

		//! \copydoc Serializable::Serialize()
		/*!
		 * \warning Serialization not implemented yet!
		 */
		void Serialize(Json::Value& json) const override
		{

		}

		//! Destructor.
		~ElasticBand() {}
	};
}
			
