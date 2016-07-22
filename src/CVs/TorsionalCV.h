#pragma once 

#include "CollectiveVariable.h"

#include <array>
#include <math.h>

namespace SSAGES
{
	// Collective variable on an proper dihedral. This will 
	// return the angle between two planes 
	// Reference: @article{VANSCHAIK1993751,
	// title = "A Structure Refinement Method Based on Molecular Dynamics in Four Spatial Dimensions",
	// journal = "Journal of Molecular Biology",
	// volume = "234",
	// number = "3",
	// pages = "751 - 762",
	// year = "1993",
	// note = "",
	// issn = "0022-2836",
	// doi = "http://dx.doi.org/10.1006/jmbi.1993.1624",
	// url = "http://www.sciencedirect.com/science/article/pii/S0022283683716244",
	// author = "René C. van Schaik and Herman J.C. Berendsen and Andrew E. Torda and Wilfred F. van Gunsteren",
	// keywords = "structure refinement",
	// keywords = "molecular dynamics: NOE"

	/*
	Blondel, A. and Karplus, M. (1996), 
	New formulation for derivatives of torsion angles and improper torsion angles in molecular mechanics: Elimination of singularities.
	J. Comput. Chem., 17: 1132–1141. doi:10.1002/(SICI)1096-987X(19960715)17:91132::AID-JCC53.0.CO;2-T
	*/

	class TorsionalCV : public CollectiveVariable
	{
	private:
		// IDs of atoms of interest.
		int _atomid1;
		int _atomid2; 
		int _atomid3;
		int _atomid4; 

		// Current value of the CV.
		double _val;

		// Use periodic boundary or not
		bool _periodic;

		// Gradients of the Dihedral CV, dtheta/dri, dtheta/drj, dtheta/drk, dtheta/drl.
		std::vector<Vector3> _grad;

		// Bounds on CV.
		std::array<double, 2> _bounds;

		// Helper function to compute the norm of a vector.
		double norm(const Vector3& v)
		{
			return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
		}

		// Helper function to compute the norm of a vector.
		double norm2(const Vector3& v)
		{
			return (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
		}

		double DotProduct(const Vector3& v, Vector3& w)
		{
			return (v[0]*w[0] + v[1]*w[1] + v[2]*w[2]);
		}

		Vector3 CrossProduct(const Vector3& u, const Vector3& v)
		{
			Vector3 Cross;
			Cross[0] = u[1]*v[2] - u[2]*v[1];
			Cross[1] = u[2]*v[0] - u[0]*v[2];
			Cross[2] = u[0]*v[1] - u[1]*v[0];

			return Cross;
		}

	public:
		// Construct an dihedral CV. The atomids specify 
		// the IDs of the atoms of interest
		// TODO: bounds needs to be an input and periodic boundary conditions
		TorsionalCV(int atomid1, int atomid2, int atomid3, int atomid4, bool periodic) : 
		_atomid1(atomid1), _atomid2(atomid2), _atomid3(atomid3), _atomid4(atomid4),
		_val(0), _periodic(periodic), _grad(0), _bounds{{0,0}}
		{
		}

		// Initialize necessary variables.
		void Initialize(const Snapshot& snapshot) override
		{
			// Initialize gradient. 
			auto n = snapshot.GetPositions().size();		
			_grad.resize(n);
		}

		// Evaluate the CV.
		void Evaluate(const Snapshot& snapshot) override
		{
			// Gradient and value. 
			const auto& pos = snapshot.GetPositions(); 
			const auto& ids = snapshot.GetAtomIDs();

			int iindex, jindex, kindex, lindex;

			iindex = jindex = kindex = lindex = -1;
			double ix = 0;
			double iy = 0;
			double iz = 0;
			double jx = 0;
			double jy = 0;
			double jz = 0;
			double kx = 0;
			double ky = 0;
			double kz = 0;
			double lx = 0;
			double ly = 0;
			double lz = 0;
			
			// Loop through atom positions
			for(size_t i = 0; i < pos.size(); ++i)
			{
				_grad[i][0] = 0;
				_grad[i][1] = 0;
				_grad[i][2] = 0;
				// If we are at the atom ID of interest, grab coordinates
				if(ids[i] == _atomid1)
				{
					//coordinates for atom i
					ix = pos[i][0];
					iy = pos[i][1];
					iz = pos[i][2];
					iindex = i;
				}
				if(ids[i] == _atomid2)
				{
					//coordinates for atom j
					jx = pos[i][0];
					jy = pos[i][1];
					jz = pos[i][2];
					jindex = i;
				}
				if(ids[i] == _atomid3)
				{
					//coordinates for atom k
					kx = pos[i][0];
					ky = pos[i][1];
					kz = pos[i][2];
					kindex = i;
				}
				if(ids[i] == _atomid4)
				{
					//coordinates for atom l
					lx = pos[i][0];
					ly = pos[i][1];
					lz = pos[i][2];
					lindex = i;
				}
			}

			if(iindex < 0 || jindex < 0 || kindex < 0 || lindex <0)
			{
				std::cout<<"Out of bounds index, could not locate an ID"<<std::endl;
				exit(0);
			}

			//Calculate pertinent vectors
			Vector3 F{{
				ix - jx,
				iy - jy,
				iz - jz}};
			Vector3 G{{
				jx - kx,
				jy - ky,
				jz - kz}};
			Vector3 H{{
				lx - kx,
				ly - ky, 
				lz - kz}};

			Vector3 rij{{
				ix - jx,
				iy - jy,
				iz - jz}};
			Vector3 rkj{{
				kx - jx,
				ky - jy,
				kz - jz}};
			Vector3 rkl{{
				lx - kx,
				ly - ky, 
				lz - kz}};

			//Calculate dihedral angle

			Vector3 rim, rln;
			double rkj2 = norm(rkj)*norm(rkj);
			for(size_t i = 0; i<3; i++)
			{
				rim[i] = rij[i] - DotProduct(rij,rkj)/(rkj2)*rkj[i];
				rln[i] = DotProduct(rkl,rkj)/(rkj2)*rkj[i] - rkl[i];
			}

			double normrim, normrln;

			normrim = norm(rim);
			normrln = norm(rln);

			auto normkj = norm(rkj);
			Vector3 rijrkjprod;
			for(size_t i = 0; i < rijrkjprod.size();i++)
				rijrkjprod[i] = rij[i]*normkj;
			auto rkjrklcross = CrossProduct(rkj, rkl);
			auto y = DotProduct(rijrkjprod, rkjrklcross);
			auto rijrkjcross = CrossProduct(rij, rkj);
			auto x = DotProduct(rijrkjcross, rkjrklcross);
			_val = atan2(y, x);

			Vector3 A = CrossProduct(F, G);
			Vector3 B = CrossProduct(H, G);

			double Zed = DotProduct(F, G)/(norm2(A)*norm(G));
			double Ned = DotProduct(H, G)/(norm2(B)*norm(G));

			for(size_t i = 0; i<3; i++)
			{
				_grad[iindex][i] = -norm(G)*A[i]/norm2(A);
				_grad[lindex][i] = norm(G)*B[i]/norm2(B);
				_grad[jindex][i] = Zed*A[i] - Ned*B[i] - _grad[iindex][i];
				_grad[kindex][i] = Ned*B[i] - Zed*A[i] - _grad[lindex][i];
			}

		}

		// Return the value of the CV.
		double GetValue() const override 
		{ 
			return _val; 
		}

		double GetPeriodicValue(double Location) const override
		{
			if(!_periodic)
				return Location;

			double pi = 3.14159;
			int n = (int)(Location/(2.0*pi));
			double PeriodicLocation = Location-2.0*n*pi;

			PeriodicLocation = Location - n*pi;
			if(PeriodicLocation < -pi)
				PeriodicLocation += 2.0*pi;
			else if (Location > pi)
				PeriodicLocation -= 2.0*pi;

			return PeriodicLocation;
		}

		// Return the gradient of the CV.
		const std::vector<Vector3>& GetGradient() const override
		{
			return _grad;
		}

		// Return the boundaries of the CV.
		const std::array<double, 2>& GetBoundaries() const override
		{
			return _bounds;
		}

		double GetDifference(const double Location) const override
		{
			double pi = 3.14159;
			double PeriodicDiff = _val - Location;

			if(!_periodic)
				return PeriodicDiff;

			PeriodicDiff = GetPeriodicValue(PeriodicDiff);

			if(PeriodicDiff > pi)
				PeriodicDiff -= 2.0*pi;
			else if(PeriodicDiff < -pi)
				PeriodicDiff += 2.0*pi;

			return PeriodicDiff;
		}
	};
}
