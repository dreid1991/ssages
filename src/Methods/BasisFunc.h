/**
 * This file is part of
 * SSAGES - Suite for Advanced Generalized Ensemble Simulations
 *
 * Copyright 2016 Joshua Moller <jmoller@uchicago.edu>
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

#include "Method.h"
#include "../CVs/CollectiveVariable.h"
#include "../Grids/Grid.h"
#include <fstream>
#include <vector>


namespace SSAGES
{

    //! Map for histogram and coefficients.
    /*!
     * A clean mapping structure for both the histogram and the coefficients.
     * All vectors are written as 1D with a row major mapping. In order to make
     * iterating easier, the mapping of the 1D vectors are written here.
     */
    struct Map
    {
        //! The coefficient value
        double value;

        //! The mapping in an array of integers
        std::vector<int> map;

        //! Constructor
        /*!
         * \param map The mapping in an array of integers.
         * \param value The coefficient value.
         */
        Map(const std::vector<int>& map,
            double value) :
            value(value), map(map)
        {}
    };

    //! Look-up table for basis functions.
	/*!
     * The structure that holds the Look-up table for the basis function. To
     * prevent repeated calculations, both the derivatives and values of the
     * Legendre polynomials is stored here. More will be added in future
     * versions.
     */
	struct BasisLUT
	{
		//! The values of the basis sets
		std::vector<double> values;

		//! The values of the derivatives of the basis sets
		std::vector<double> derivs;

        //! Constructor.
        /*!
         * \param values The values of the basis sets.
         * \param derivs The values of the derivatives of the basis sets.
         */
		BasisLUT(const std::vector<double>& values,
			const std::vector<double>& derivs) :
			values(values), derivs(derivs)
		{}
	};
		
    //! Basis Function Sampling Algorithm
    /*!
     * \ingroup Methods
     *
     * Implementation of the Basis Function Sampling Method based on
     * \cite WHITMER2014190602.
     */
	class Basis : public Method
	{
	private:	
        
        //! Histogram of visited states.
        /*!
         * Histogram is stored locally. It is a 1D vector that holds N
         * dimensional data over the number of walkers using a row major
         * mapping.
         */
        std::vector<Map> _hist;

        //! Locally defined histogram array for easy mpi operations.
        /*!
         * \note It does take up more memory.
         */
        std::vector<int> _histlocal;

        //! Globally defined histogram array for easy mpi operations.
        /*!
         * \note Needs lots of memory.
         */
        std::vector<int> _histglobal;

        //! Globally located coefficient values.
		/*!
         * As coefficients are updated at the same time, the coefficients
         * should only be defined globally.
         */
		std::vector<Map> _coeff;

        //! The biased histogram of states.
        /*!
         * The biased histogram of states has the form _hist*exp(phi*beta),
         * where phi is the bias potential and beta is the inverse of the
         * temperature. It is defined globally.
         */
        std::vector<double> _unbias;

        //! The coefficient array for restart runs
        std::vector<double> _coeff_arr;

        //! The Basis set lookup table, also defined globally
		std::vector<BasisLUT> _LUT;

        //! Derivatives of the bias potential
        /*!
         * The derivatives of the bias potential imposed on the system.
         * They are evaluated by using the lookup table.
         */
		std::vector<double> _derivatives;

        //! The order of the basis polynomials
        std::vector<unsigned int> _polyords;

        //! Storing number of bins for simplicity and readability of code
        std::vector<unsigned int> _nbins;

        //! Spring constants for restrained system.
        /*!
         * The system uses this to determine if the system is to be restrained
         * on the defined interval. The user inputs the spring constants if the
         * system is not periodic.
         */
        std::vector<double> _restraint;

        //! Upper position of the spring restraint.
        std::vector<double> _boundUp;

        //! Lower position of the spring restraint.
        std::vector<double> _boundLow;
        
        //! Frequency of coefficient updates
		unsigned int _cyclefreq;
        
        //! The node that the current system belongs to, primarily for printing and debugging.
        unsigned int _mpiid;

        //! Weighting for potentially faster sampling.
        /*!
         * Weighting can be used to potentially sample faster, however it can
         * cause the simulation to explode. By default this value will be set
         * to 1.0
         */
        double _weight;

        //! Self-defined temperature.
        /*!
         * In the case of the MD engine using a poorly defined temperature, the
         * option to throw it into the method is available. If not provided it
         * takes the value from the engine.
         */
        double _temperature;

        //! The tolerance criteria for the system .
        double _tol;

        //! A variable to check to see if the simulation is in bounds or not.
        bool _bounds;

        //! A check to see if you want the system to end when it reaches the convergence criteria.
        bool _converge_exit;

		//! Updates the bias projection of the PMF.
        /*!
         * \param cvs List of collective variables.
         * \param beta Temperature equivalent.
         */
		void UpdateBias(const CVList& cvs, const double);

		//! Computes the bias force.
        /*!
         * \param cvs List of collective variables.
         */
		void CalcBiasForce(const CVList& cvs);

		//! Prints the current bias to a defined file from the JSON.
        /*!
         * \param cvs List of collective variables.
         */
		void PrintBias(const CVList& cvs, const double);

        //! Initializes the look up tables for polynomials
        /*!
         * \param cvs List of collective variables.
         */
        void BasisInit(const CVList& cvs);

		//! Output stream for basis projection data.
		std::ofstream _basisout;

        //! Output stream for coefficients (for reading purposes)
        std::ofstream _coeffout;

        //! The option to name both the basis and coefficient files will be given
        //! Basis filename 
        std::string _bnme;

        //! Coefficient filename
        std::string _cnme;


	public:
        //! Constructor
		/*!
         * \param world MPI global communicator.
         * \param comm MPI local communicator.
         * \param polyord Order of Legendre polynomials.
         * \param restraint Restraint spring constants.
         * \param boundUp Upper bounds of restraint springs.
         * \param boundLow Lower bounds of restraint springs.
         * \param cyclefreq Cycle frequency.
         * \param frequency Frequency with which this Method is applied.
         * \param bnme Basis file name.
         * \param cnme Coefficient file name.
         * \param temperature Automatically set temperature.
         * \param tol Threshold for tolerance criterion.
         * \param weight Weight for improved sampling.
         * \param converge If \c True quit on convergence.
         *
         * Constructs an instance of the Basis function sampling method. The
         * coefficients describes the basis projection of the system. This is
         * updated once every _cyclefreq. For now, only the Legendre polynomial
         * is implemented. Others will be added later.
         */
		Basis(boost::mpi::communicator& world,
			 boost::mpi::communicator& comm,
			 const std::vector<unsigned int>& polyord,
             const std::vector<double>& restraint,
             const std::vector<double>& boundUp,
             const std::vector<double>& boundLow,
             unsigned int cyclefreq,
			 unsigned int frequency,
             const std::string bnme,
             const std::string cnme,
             const double temperature,
             const double tol,
             const double weight,
             bool converge) : 
		Method(frequency, world, comm), _hist(), _histlocal(), _histglobal(),
        _coeff(), _unbias(), _coeff_arr(), _LUT(), _derivatives(), _polyords(polyord),
        _nbins(), _restraint(restraint), _boundUp(boundUp), _boundLow(boundLow),
        _cyclefreq(cyclefreq), _mpiid(0), _weight(weight),
        _temperature(temperature), _tol(tol),
        _converge_exit(converge), _bnme(bnme), _cnme(cnme)
		{
		}

		//! Pre-simulation hook.
        /*!
         * \param snapshot Simulation snapshot.
         * \param cvs List of CVs.
         */
		void PreSimulation(Snapshot* snapshot, const CVList& cvs) override;

		//! Post-integration hook.
        /*!
         * \param snapshot Simulation snapshot.
         * \param cvs List of CVs.
         */
		void PostIntegration(Snapshot* snapshot, const CVList& cvs) override;

		//! Post-simulation hook.
        /*!
         * \param snapshot Simulation snapshot.
         * \param cvs List of CVs.
         */
		void PostSimulation(Snapshot* snapshot, const CVList& cvs) override;

        //! \copydoc Serializable::Serialize()
        /*!
         * \warning Serialization is not implemented yet!
         */
        void SetIteration(const int iter)
        {
            _iteration = iter;
        }

        void SetBasis(const std::vector<double>&coeff, std::vector<double>&unbias)
        {
            _coeff_arr = coeff;
            _unbias = unbias;
        }

		void Serialize(Json::Value& json) const override
		{
            json["type"] = "Basis";
            for(auto& p: _polyords)
                json["CV_coefficients"].append(p);

            for(auto& k: _restraint)
                json["CV_restraint_spring_constants"].append(k);

            for(auto& u: _boundUp)
                json["CV_restraint_maximums"].append(u);

            for(auto& l: _boundLow)
                json["CV_restraint_minimums"].append(l);

            for(auto& b: _unbias)
                json["bias_hist"].append(b);

            for(auto& c: _coeff_arr)
                json["coefficients"].append(c);

            json["tolerance"] = _tol;

            json["convergence_exit"] = _converge_exit;

            json["basis_filename"] = _bnme;

            json["coeff_filename"] = _cnme;

            json["iteration"] = _iteration;

            json["cycle_frequency"] = _cyclefreq;

            json["weight"] = _weight;

            json["temperature"] = _temperature;
		}

        //! Destructor.
		~Basis() {}
	};
}
			
