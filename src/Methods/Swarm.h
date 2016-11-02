/**        
- * This file is part of        
- * SSAGES - Suite for Advanced Generalized Ensemble Simulations        
- *     
- * Copyright 2016 Cody Bezik <bezik@uchicago.edu>      
- *     
- * SSAGES is free software: you can redistribute it and/or modify      
- * it under the terms of the GNU General Public License as published by        
- * the Free Software Foundation, either version 3 of the License, or       
- * (at your option) any later version.     
- *     
- * SSAGES is distributed in the hope that it will be useful,       
- * but WITHOUT ANY WARRANTY; without even the implied warranty of      
- * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
- * GNU General Public License for more details.        
- *     
- * You should have received a copy of the GNU General Public License       
- * along with SSAGES.  If not, see <http://www.gnu.org/licenses/>.     
- */
#include "StringMethod.h"
#include "../CVs/CollectiveVariable.h"
#include <fstream>
#include <iostream>

namespace SSAGES
{
    //! Swarm of Trajectories String Method
    /*!
     * \ingroup Methods
     *
     * Implementation of the swarm of trajectories string method.
     */
    class Swarm : public StringMethod //Calls Method constructor first, then Swarm constructor
    {
        private:

            //! Drift of CVs for one iteration
            std::vector<double> _cv_drift; 
                        
            //! Total number of MD steps for initialization for one iteration
            unsigned int _initialize_steps; 

            //! Length to run before harvesting a trajectory for unrestrained sampling
            unsigned int _harvest_length;

            //! Total number of restrained MD steps for one iteration
            unsigned int _restrained_steps; 

            //! Number of trajectories per swarm
            unsigned int _number_trajectories;

            //! Length of unrestrained trajectories
            unsigned int _swarm_length;

            //! Total number of unrestrained MD steps for one iteration
            unsigned int _unrestrained_steps;

            //! For indexing trajectory vectors
            int _index; 

            //! Store atom IDs for starting trajecotires
            std::vector<Label> _traj_atomids; 

            //! Updates the positions of the string
            void StringUpdate();

            //! Helper function check if CVs are initialized correctly
            bool CVInitialized(const CVList& cvs);

            //! Flag for determing whether to perform initialization or not
            bool sampling_started;
            
            //! Flag for whether a snapshot was stored in the umbrella sampling
            bool snapshot_stored;        
            
            //! Flag for whether a system is initialized at a given iteration
            bool initialized;

            //! Flag for whether a system was initialized before it checked whether other systems were
            bool original_initialized; 

        public:

            //! Constructor.
            /*!
             * \param world MPI global communicator.
             * \param com MPI local communicator.
             * \param centers List of centers.
             * \param NumNodes number of nodes.
             * \param spring Spring constant.
             * \param frequency Aplly method with this frequency.
             * \param InitialSteps Number of initial steps.
             * \param HarvestLength Length of trajectory before weighing.
             * \param NumberTrajectories Number of trajectories.
             * \param SwarmLength Lengt of the swarms.
             *
             * Constructs an instance of the swarm of trajectories method.
             */
            Swarm(boost::mpi::communicator& world, 
                    boost::mpi::communicator& comm, 
                    const std::vector<double>& centers,
                    unsigned int maxiterations,
                    const std::vector<double> cvspring,
                    unsigned int frequency,
                    unsigned int InitialSteps, 
                    unsigned int HarvestLength, 
                    unsigned int NumberTrajectories, 
                    unsigned int SwarmLength) : 
                StringMethod(world, comm, centers, maxiterations, cvspring, frequency),                 
                _cv_drift(), 
                _initialize_steps(InitialSteps), 
                _harvest_length(HarvestLength), 
                _number_trajectories(NumberTrajectories), 
                _swarm_length(SwarmLength)
        {
            _cv_drift.resize(_centers.size(), 0);
            _prev_positions.resize(_number_trajectories);
            _prev_velocities.resize(_number_trajectories);
            _prev_IDs.resize(_number_trajectories);
            //Additional initializing

            _index = 0;  
            _restrained_steps = _harvest_length*_number_trajectories; 
            _unrestrained_steps = _swarm_length*_number_trajectories;
            sampling_started = false;
            snapshot_stored = false;

            _iterator = 0; //Override default StringMethod.h initializing
        }

            //! Post-integration hook
            void PostIntegration(Snapshot* snapshot, const CVList& cvs) override; 

            void Serialize(Json::Value& json) const override
            {
                StringMethod::Serialize(json);

                json["flavor"] = "SWARM";

                json["initial_steps"] = _initialize_steps;
                json["harvest_length"] = _harvest_length;
                json["number_of_trajectories"] = _number_trajectories;
                json["swarm_length"] = _swarm_length; 
            }

            //! Destructor
            ~Swarm()
            {

            }
    };
}

